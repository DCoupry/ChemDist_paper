
"""
Copyright 2021 GlaxoSmithKline Research & Development Limited

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import pandas
import argparse
import numpy
import torch
from tqdm.auto import tqdm
from rdkit.Chem import MolFromSmiles, AllChem
from rdkit.DataStructs import FingerprintSimilarity
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from dgllife.model import MPNNPredictor
from chemdist import DistanceNetworkLigthning
import logging

MODEL = DistanceNetworkLigthning.load_from_checkpoint("model_ablation.pt")._net
NF = CanonicalAtomFeaturizer()
BF = CanonicalBondFeaturizer()


def chemdist_func(s):
    g = smiles_to_bigraph(smiles=s, node_featurizer=NF, edge_featurizer=BF)
    nfeats = g.ndata.pop('h')
    efeats = g.edata.pop('e')
    membed = MODEL(g, nfeats, efeats).cpu().detach().numpy().ravel()
    return membed


def similarities_chemdist(data):
    fp_anchor = numpy.stack(data.anchor.progress_apply(chemdist_func))
    fp_positive = numpy.stack(data.positive.progress_apply(chemdist_func))
    fp_negative = numpy.stack(data.negative.progress_apply(chemdist_func))
    sim_ap = numpy.linalg.norm(fp_anchor - fp_positive, axis=-1)
    sim_an = numpy.linalg.norm(fp_anchor - fp_negative, axis=-1)
    return sim_ap, sim_an


def similarities_ecfp4(data_points):
    mol_anchor = MolFromSmiles(data_points.anchor)
    mol_positive = MolFromSmiles(data_points.positive)
    mol_negative = MolFromSmiles(data_points.negative)
    fp_anchor = AllChem.GetMorganFingerprintAsBitVect(mol_anchor, 2)
    fp_positive = AllChem.GetMorganFingerprintAsBitVect(mol_positive, 2)
    fp_negative = AllChem.GetMorganFingerprintAsBitVect(mol_negative, 2)
    sim_ap = FingerprintSimilarity(fp_anchor, fp_positive)
    sim_an = FingerprintSimilarity(fp_anchor, fp_negative)
    return sim_ap, sim_an


def plot(sims):
    import matplotlib.pyplot as plt
    sims = sims.rename(columns={"pos_tanimoto": "A-P ECFP4", 
                                "neg_tanimoto": "A-N ECFP4", 
                                "pos_dist": "A-P MPNN embedding", 
                                "neg_dist": "A-N MPNN embedding",}) 
    sims["A-P ECFP4"] = 1 - sims["A-P ECFP4"]
    sims["A-N ECFP4"] = 1 - sims["A-N ECFP4"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    sims["A-P ECFP4"].plot.kde(linestyle="solid", color="k", ax=axes[0])
    sims["A-N ECFP4"].plot.kde(linestyle="dotted", color="k", ax=axes[0])
    sims["A-P MPNN embedding"].plot.kde(linestyle="solid", color="k", ax=axes[1]) 
    sims["A-N MPNN embedding"].plot.kde(linestyle="dotted", color="k", ax=axes[1])
    axes[0].set_xlabel("Tanimoto dissimilarity")
    axes[1].set_xlabel("Euclidean distance")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    axes[0].spines["top"].set_visible(False)
    axes[1].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[0].set_xlim(0,1)
    axes[1].set_xlim(0, min(sims["A-N MPNN embedding"].max(), sims["A-P MPNN embedding"].max()))
    plt.savefig("sims.svg")
    return


def main(args):
    df = pandas.read_csv(args.data).sample(frac=args.subsample)
    df = df.dropna()
    logging.info("Computing ECFP4 similarities...")
    sims_ecfp4 = numpy.array([similarities_ecfp4(dp)
                             for _, dp in tqdm(df.iterrows(), total=len(df))])
    logging.info("Computing Chemdist distances...")
    sims_chemdist = numpy.stack(similarities_chemdist(df), axis=-1)
    sims = numpy.concatenate([sims_ecfp4, sims_chemdist], axis=-1)
    sims = pandas.DataFrame(sims, columns=["pos_tanimoto", "neg_tanimoto",
                                           "pos_dist", "neg_dist",])
    print(sims)
    sims.to_csv("sims.csv")
    plot(sims)
    return


if __name__ == "__main__":
    tqdm.pandas()
    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", type=str)
    parser.add_argument("-s", "--subsample", type=float)
    args = parser.parse_args()
    main(args)
