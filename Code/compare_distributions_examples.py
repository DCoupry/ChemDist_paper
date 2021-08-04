
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
import scipy
import torch
from tqdm.auto import tqdm
from rdkit.Chem import MolFromSmiles, AllChem, AddHs, RemoveHs, MolToSmiles
from rdkit.DataStructs import BulkTanimotoSimilarity
from dgllife.utils import mol_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from chemdist import DistanceNetworkLigthning

MODEL = DistanceNetworkLigthning.load_from_checkpoint("model_ablation.pt")._net.cuda()
NF = CanonicalAtomFeaturizer()
BF = CanonicalBondFeaturizer()
REFERENCES = [r"OC(=O)C1=CSC=N1",
              r"NC1=NC(=CS1)C(O)=O",
              r"OC(=O)C1=CSC(NC(Cl)=O)=N1",
              r"CC(C)C(Cl)C(=O)NC1=NC(=CS1)C(O)=O",
              r"OC(=O)C1=CSC(NC(=O)CC2=CN=CC(Cl)=C2)=N1",
              r"CC(C)OC1=CC(=CN=C1CC(F)(F)F)C1(CN(C)CC(C)=C1)C(=O)NC1=NC2=C(CC(C)\C=C\COC3CC4(CCC3C(=O)O4)COC2=O)S1"]


def chemdist_embed(m):
    g = mol_to_bigraph(mol=m, node_featurizer=NF, edge_featurizer=BF).to("cuda")
    nfeats = g.ndata.pop('h')
    efeats = g.edata.pop('e')
    return MODEL(g, nfeats, efeats).cpu().detach().numpy().ravel()

def chemdist(candidates, target):
    embeddings = []
    for m in tqdm(candidates, leave=False):
        embeddings.append(chemdist_embed(m))
    embeddings = numpy.stack(embeddings)
    target_embedding = chemdist_embed(target)
    dists = numpy.linalg.norm(embeddings - target_embedding, axis=1)
    return dists


def main(args):
    df = pandas.read_csv(args.data, index_col="index").sample(frac=args.subsample)
    df = df.dropna()
    scores = pandas.DataFrame(index=df.index.values, columns=REFERENCES)
    mols = df.SMILES.progress_apply(MolFromSmiles)
    for ref in tqdm(REFERENCES):
        molref = MolFromSmiles(ref)
        scores[ref] = chemdist(mols, molref)
    scores.to_csv("references_scores.csv", index=False)
    print(scores)
    return

if __name__ == "__main__":
    tqdm.pandas()
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", type=str)
    parser.add_argument("-s", "--subsample", type=float)
    args = parser.parse_args()
    main(args)
