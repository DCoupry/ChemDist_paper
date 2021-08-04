
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
import argparse
import logging
import math
import os
import warnings

import dgl
import dill
import numpy
import pandas
import pytorch_lightning
import rdkit.Chem
import torch
from dgllife.model import MPNNPredictor
from dgllife.utils import (CanonicalAtomFeaturizer, CanonicalBondFeaturizer,
                           smiles_to_bigraph)
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from tqdm.auto import tqdm

pytorch_lightning.seed_everything(42)
warnings.filterwarnings("ignore")

BOND_FEAT_SIZE = 12
ATOM_FEAT_SIZE = 74

def ablation(g, ep=0.05, np=0.01):
    nodes_todel = []
    for n in range(g.number_of_nodes()):
        if numpy.random.rand() < 0.01:
            nodes_todel.append(n)
    g = dgl.remove_nodes(g, nodes_todel)
    # 5% chance of deleting an edge
    edges_todel = []
    for n in range(g.number_of_edges()):
        if numpy.random.rand() < 0.05:
            edges_todel.append(n)
    g = dgl.remove_edges(g, edges_todel)
    return g

def graph_collate_fn(data):
    bg = []
    for g in data:
        # MAKE SOME NOISE!! WHOOP WHOOP!
        # 1% chance of deleting a node
        g = ablation(g)
        bg.append(g)
    bg = dgl.batch(bg)
    # base init for unknown
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    node_feats = bg.ndata.pop('h')
    edge_feats = bg.edata.pop('e')
    return bg, node_feats, edge_feats


def triplet_collate_fn(data):
    a, p, n = zip(*data)
    m_a = [i for i,v in enumerate(a) if v == None]
    m_p = [i for i,v in enumerate(p) if v == None]
    m_n = [i for i,v in enumerate(n) if v == None]
    m = set(m_a + m_p + m_n)
    a = [v for i,v in enumerate(a) if i not in m]
    p = [v for i,v in enumerate(p) if i not in m]
    n = [v for i,v in enumerate(n) if i not in m]
    a = graph_collate_fn(a)
    p = graph_collate_fn(p)
    n = graph_collate_fn(n)
    return a, p, n


class TripletDataset(torch.utils.data.Dataset):

    def __init__(self, anchors, positives, negatives):
        feat_kwargs = {"node_featurizer": CanonicalAtomFeaturizer(),
                       "edge_featurizer": CanonicalBondFeaturizer(),
                       "add_self_loop": False}
        self.anchors = [smiles_to_bigraph(a, **feat_kwargs) for a in tqdm(anchors)]
        self.positives = [smiles_to_bigraph(p, **feat_kwargs) for p in tqdm(positives)]
        self.negatives = [smiles_to_bigraph(n, **feat_kwargs) for n in tqdm(negatives)]

        
    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx):
        a = self.anchors[idx]
        p = self.positives[idx]
        n = self.negatives[idx]
        return a, p, n



class TripletDataModule(pytorch_lightning.LightningDataModule):

    def __init__(self, datapath, batch_size, subsample):
        super().__init__()
        self.datapath = datapath
        self.subsample = subsample
        self.dl_kwargs = {"batch_size": batch_size, 
                          "collate_fn": triplet_collate_fn, 
                          "num_workers": 0,
                          "pin_memory":True}


    def prepare_data(self): 
        if not os.path.isfile("triplets.bin"):
            data = pandas.read_csv(self.datapath).sample(frac=min(1.0, self.subsample))
            triplets = TripletDataset(anchors=data.anchor.values, 
                                      positives=data.positive.values , 
                                      negatives=data.negative.values)
            with open("triplets.bin", "wb") as binfile:
                dill.dump(triplets, binfile)
        return

    def setup(self, stage):
        with open("triplets.bin", "rb") as binfile:
            triplets = dill.load(binfile)
        indices = numpy.arange(len(triplets))
        numpy.random.shuffle(indices)
        sep08, sep09 = int(len(indices)*0.8), int(len(indices)*0.9)
        if stage == 'fit' or stage is None:
            train_indices = indices[:sep08]
            valid_indices = indices[sep08:sep09]
            self.train_ds = torch.utils.data.Subset(triplets, train_indices)
            self.valid_ds = torch.utils.data.Subset(triplets, valid_indices)
        if stage == 'test' or stage is None:
            test_indices = indices[sep09:]
            self.test_ds = torch.utils.data.Subset(triplets, test_indices)
        return

    def train_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.train_ds, **self.dl_kwargs)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.valid_ds, **self.dl_kwargs)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.test_ds, **self.dl_kwargs)


class DistanceNetworkLigthning(pytorch_lightning.LightningModule):

    def __init__(self, embed_size, node_in_feats, edge_in_feats):
        super().__init__()
        self.save_hyperparameters()
        self._net = MPNNPredictor(node_in_feats=node_in_feats,
                                  edge_in_feats=edge_in_feats,
                                  n_tasks=embed_size)
        self._loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)
        return None

    def forward(self, anchor, positive, negative):
        anchor_out = self._net(*anchor)
        positive_out = self._net(*positive)
        negative_out = self._net(*negative)
        return anchor_out, positive_out, negative_out

    def training_step(self, batch, batch_nb):
        anchor, positive, negative = batch
        anchor_out, positive_out, negative_out = self.forward(anchor, positive, negative)
        loss = self._loss(anchor_out, positive_out, negative_out)
        self.log('train_loss', loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_nb):
        anchor, positive, negative = batch
        anchor_out, positive_out, negative_out = self.forward(anchor, positive, negative)
        loss = self._loss(anchor_out, positive_out, negative_out)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_nb): 
        anchor, positive, negative = batch
        anchor_out, positive_out, negative_out = self.forward(anchor, positive, negative)
        loss = self._loss(anchor_out, positive_out, negative_out)
        self.log('test_loss', loss, sync_dist=True)
        return loss

    def configure_optimizers(self):  
        opt = torch.optim.Adam(self.parameters(), lr=1.0e-3)
        return opt


def main(args):
    early_stopping = EarlyStopping(mode="min", monitor="val_loss", patience=25)
    model_checkpoint = ModelCheckpoint(monitor="val_loss", mode="min")
    data = TripletDataModule(datapath=args.data, 
                             batch_size=args.batch_size,
                             subsample=args.subsample)
    model = DistanceNetworkLigthning(embed_size=args.embed_size, 
                                     node_in_feats=ATOM_FEAT_SIZE,
                                     edge_in_feats=BOND_FEAT_SIZE)
    trainer = Trainer(gpus=1,
                      default_root_dir="training",
                      callbacks=[early_stopping, model_checkpoint], 
                      logger=True,
		      resume_from_checkpoint="restart.ckpt",
                      weights_summary="full")
    trainer.fit(model, data)
    best_model = model._net.cpu().eval()
    test_score = trainer.test(datamodule=data, verbose=True)
    torch.save(best_model.state_dict(), f"best_{args.embed_size}.pt")
    with open("test_scores.txt", "w") as test_file:
        txt = "\n".join([f"{k} : {v:.5f}" for k, v in test_score[0].items()])
        test_file.write(txt)
    return None


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", type=int)
    parser.add_argument("-e", "--embed_size", type=int)
    parser.add_argument("-d", "--data", type=str)
    parser.add_argument("-s", "--subsample", default=1.0, type=float)
    args = parser.parse_args()
    main(args)
