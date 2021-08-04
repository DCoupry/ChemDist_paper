
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
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem import MolFromSmiles
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, cohen_kappa_score
from sklearn.neighbors import KNeighborsClassifier
import numpy
import torch
from lightgbm import LGBMClassifier
from tqdm.auto import tqdm
from chemdist import DistanceNetworkLigthning
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from mlxtend.evaluate import mcnemar, mcnemar_table, cochrans_q

MODEL = DistanceNetworkLigthning.load_from_checkpoint("model_ablation.pt")._net
NF = CanonicalAtomFeaturizer()
BF = CanonicalBondFeaturizer()

def chemdist_func(s):
    g = smiles_to_bigraph(smiles=s, node_featurizer=NF, edge_featurizer=BF)
    nfeats = g.ndata.pop('h')
    efeats = g.edata.pop('e')
    return MODEL(g, nfeats, efeats).cpu().detach().numpy().ravel()


def get_score(data):
    ps = []
    aucs_ecfp0 = []
    aucs_ecfp4 = []
    aucs_cd = []
    X_ecfp4 = numpy.stack(data.index.map(lambda s: GetMorganFingerprintAsBitVect(MolFromSmiles(s), 2, nBits=1024)))
    X_ecfp0 = numpy.stack(data.index.map(lambda s: GetMorganFingerprintAsBitVect(MolFromSmiles(s), 0, nBits=1024)))
    X_cdist = numpy.stack(data.index.map(chemdist_func))
    y = data.IS_ACTIVE.values
    rkf = StratifiedShuffleSplit(n_splits=5)
    for train_idx, test_idx in tqdm(rkf.split(X_ecfp4, y=y), total=5, leave=False):
        y_train, y_test = y[train_idx], y[test_idx]
        X_ecfp4_train, X_ecfp4_test = X_ecfp4[train_idx], X_ecfp4[test_idx]
        X_ecfp0_train, X_ecfp0_test = X_ecfp0[train_idx], X_ecfp0[test_idx]
        X_cdist_train, X_cdist_test = X_cdist[train_idx], X_cdist[test_idx]
        knn_ecfp4 = KNeighborsClassifier(n_neighbors=3, weights="distance", metric="jaccard", n_jobs=32)
        knn_ecfp0 = KNeighborsClassifier(n_neighbors=3, weights="distance", metric="jaccard", n_jobs=32)
        knn_cdist = KNeighborsClassifier(n_neighbors=3, weights="distance", metric="euclidean", n_jobs=32)
        knn_ecfp4.fit(X_ecfp4_train, y_train)
        knn_ecfp0.fit(X_ecfp0_train, y_train)
        knn_cdist.fit(X_cdist_train, y_train)
        y_ecfp4 = knn_ecfp4.predict(X_ecfp4_test)
        y_ecfp0 = knn_ecfp0.predict(X_ecfp0_test)
        y_cdist = knn_cdist.predict(X_cdist_test)
        auc_ecfp0 = cohen_kappa_score(y_test, y_ecfp0)
        auc_ecfp4 = cohen_kappa_score(y_test, y_ecfp4)
        auc_cd = cohen_kappa_score(y_test, y_cdist)
        Q, p = cochrans_q(y_test, y_ecfp0, y_ecfp4, y_cdist)
        ps.append(p)
        aucs_ecfp0.append(auc_ecfp0)
        aucs_ecfp4.append(auc_ecfp4)
        aucs_cd.append(auc_cd)
    return numpy.mean(ps), numpy.mean(aucs_cd), numpy.mean(aucs_ecfp4), numpy.mean(aucs_ecfp0)


def main(args):
    df = pandas.read_csv(args.data)
    # df = df[df.target.isin(list(set(df.target))[:5])]
    df = df.set_index("SMILES")
    output = pandas.DataFrame(index=list(set(df.target)),
                              columns=["Cochrans p-val", 
                                       "Kappa(Chemdist)", 
                                       "Kappa(ECFP4)", 
                                       "Kappa(ECFP0)"])
    groups = df.groupby("target")
    pbar = tqdm(groups)
    for gname, g in pbar:
       try:
            ps, aucs_cd, aucs_ecfp4, aucs_ecfp0 = get_score(g)
            pbar.set_description(f"BASE {aucs_ecfp0:.2f} | ECFP4 {aucs_ecfp4:.2f} | CHEMDIST {aucs_cd:.2f}")
            output.loc[gname, "Cochrans p-val"] = ps
            output.loc[gname, "Kappa(Chemdist)"] = aucs_cd
            output.loc[gname, "Kappa(ECFP4)"] = aucs_ecfp4
            output.loc[gname, "Kappa(ECFP0)"] = aucs_ecfp0
       except Exception:
           continue
    print(output)
    output.to_csv(args.output)
    # odf.plot.box(notch=True)
    # plt.savefig("box.svg")
    return

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--data', type=str, required=True,
                        help='path to an input csv file, with a SMILES, PIC50 and target columns')
    parser.add_argument('-o','--output', default="Kappa_Cochran.csv", type=str,
                        help='Path of output csv file containing ROC-AUC scores and p-values')
    args = parser.parse_args()
    main(args)
