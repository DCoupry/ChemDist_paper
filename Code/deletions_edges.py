
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
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from chemdist import DistanceNetworkLigthning


MODEL = DistanceNetworkLigthning.load_from_checkpoint("model_ablation.pt")._net

def chemdist_func(s, p=0.0):
    g = smiles_to_bigraph(smiles=s, 
                          node_featurizer=CanonicalAtomFeaturizer(), 
                          edge_featurizer=CanonicalBondFeaturizer())
    nbonds = g.number_of_edges()
    todel = []
    for n in range(nbonds):
        if numpy.random.rand() < p:
            todel.append(n)
    g.remove_edges(todel)
    nfeats = g.ndata.pop('h')
    efeats = g.edata.pop('e')
    membed = MODEL(g, nfeats, efeats).cpu().detach().numpy().ravel()
    return membed

def plot(df):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(1,1, figsize=(7,7))
    df = df.rename(columns={"Learned embedding distance": "Dataset average"})
    df.plot("P(edge drop)", "Dataset average", color="k", ax=ax)
    ax.fill_between(df["P(edge drop)"], df["Q25"], df["Q75"], alpha=0.2, color="k", label="50% CI")
    ax.legend(frameon=False, loc=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("edge drop probability")
    ax.set_ylabel("Embedding distance")
    ax.set_xlim(0,0.5)
    ax.set_ylim(0,df["Q75"].max())
    plt.savefig("deletions.pdf")
    plt.show()
    return

def main(args):
    df = pandas.read_csv(args.data, index_col="index").sample(frac=args.subsample)
    df = df.dropna()
    ref = df.SMILES.progress_apply(lambda s: chemdist_func(s, p=0.0))
    drop_probas = numpy.linspace(0.0, 0.5, 50)
    for p in drop_probas:
        this_p = df.SMILES.progress_apply(lambda s: chemdist_func(s, p=p))
        df[f"P={p:.3f}"] = (ref - this_p).apply(numpy.linalg.norm)
    odf = pandas.concat([df.mean(), df.quantile(0.25), df.quantile(0.75)], axis=1)
    odf.columns = ["Learned embedding distance", "Q25", "Q75"]
    odf["P(edge drop)"] = drop_probas
    print(odf)
    odf.to_csv("deletions_results_ablation_edges.csv")
    return


if __name__ == "__main__":
    tqdm.pandas()
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", type=str)
    parser.add_argument("-s", "--subsample", type=float)
    args = parser.parse_args()
    main(args)
