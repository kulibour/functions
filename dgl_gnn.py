import os
import numpy as np
import pandas as pd
import sys
import argparse
import copy
from joblib import Parallel, delayed

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem,MolFromSmiles,GetPeriodicTable,rdDepictor
from rdkit.Chem.Descriptors import ExactMolWt,NumHAcceptors,NumHDonors,MolLogP
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdmolops import AddHs###加H

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import dgl
from dgl.nn.pytorch import NNConv
from dgl.nn.pytorch import Set2Set
cpu_num = 1
os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num)

ATOM_TYPES = ['N','C','O','S','F','Cl','P','Se','Br','I','B','Si']

CHIRALITY = [rdkit.Chem.rdchem.ChiralType.CHI_OTHER,#手性类型 
             rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
             rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
             rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED]


HYBRIDIZATION = [rdkit.Chem.rdchem.HybridizationType.OTHER,#轨道杂化类型
                 rdkit.Chem.rdchem.HybridizationType.S,
                 rdkit.Chem.rdchem.HybridizationType.SP,
                 rdkit.Chem.rdchem.HybridizationType.SP2,
                 rdkit.Chem.rdchem.HybridizationType.SP3,
                 rdkit.Chem.rdchem.HybridizationType.SP3D,
                 rdkit.Chem.rdchem.HybridizationType.SP3D2,
                 rdkit.Chem.rdchem.HybridizationType.UNSPECIFIED]


BOND_TYPES = [rdkit.Chem.rdchem.BondType.SINGLE,#键的类型
              rdkit.Chem.rdchem.BondType.DOUBLE,
              rdkit.Chem.rdchem.BondType.TRIPLE,
              rdkit.Chem.rdchem.BondType.AROMATIC]

BOND_STEREO = [rdkit.Chem.rdchem.BondStereo.STEREONONE,#键的顺反
               rdkit.Chem.rdchem.BondStereo.STEREOANY,
               rdkit.Chem.rdchem.BondStereo.STEREOZ,
               rdkit.Chem.rdchem.BondStereo.STEREOE,]

def mol_to_graph(mol):###将化合物转为DGL中的graph数据，并添加原子和键的特征
    begin_idxs = [bond.GetBeginAtomIdx() for bond in mol.GetBonds()]
    end_idxs = [bond.GetEndAtomIdx() for bond in mol.GetBonds()]
    g = dgl.graph((begin_idxs,end_idxs),num_nodes=mol.GetNumAtoms())
    ###ATOM features
    atom_features = []
    periodic_table = Chem.GetPeriodicTable()
    for atom in mol.GetAtoms():
        # atom = mol.GetAtoms()[0]
        atom_feat = []
        atom_type = [0] * (len(ATOM_TYPES)+1)###原子类型
        if atom.GetSymbol() in ATOM_TYPES:
            atom_type[ATOM_TYPES.index(atom.GetSymbol())] = 1
        else:
            atom_type[-1] = 1
        chiral = [0] * len(CHIRALITY)###手性类型
        chiral[CHIRALITY.index(atom.GetChiralTag())] = 1
        ex_valence = atom.GetExplicitValence()###显式化合价
        charge = atom.GetFormalCharge()###电荷
        hybrid = [0] * len(HYBRIDIZATION)#轨道杂化类型
        hybrid[HYBRIDIZATION.index(atom.GetHybridization())] = 1
        degree = atom.GetDegree()###度
        valence = atom.GetImplicitValence()###隐藏的H原子数
        aromatic = int(atom.GetIsAromatic())###是否为芳香族原子
        ex_hs = atom.GetNumExplicitHs()###显式H原子数
        im_hs = atom.GetNumImplicitHs()###隐式H原子数
        rad = atom.GetNumRadicalElectrons()###自由基电子数
        ring = int(atom.IsInRing())###是否在环中
        mass = periodic_table.GetAtomicWeight(atom.GetSymbol())###质量
        vdw = periodic_table.GetRvdw(atom.GetSymbol())###范德华半径
        atom_feat = atom_type+chiral+[ex_valence,charge]+hybrid+[degree,valence,aromatic,ex_hs,im_hs,rad,ring,mass,vdw]
        atom_features.append(atom_feat)
    ###BOND features
    bond_features = []
    for bond in mol.GetBonds():
        bond_feat = []
        bond_type = [0] * len(BOND_TYPES)#键的类型
        bond_type[BOND_TYPES.index(bond.GetBondType())] = 1
        bond_stereo = [0] * len(BOND_STEREO)##键的顺反
        bond_stereo[BOND_STEREO.index(bond.GetStereo())] = 1
        isConjugated = float(bond.GetIsConjugated())###是否共轭
        isinRing = float(bond.IsInRing())###是否在环中
        bond_feat = bond_type+bond_stereo+[isConjugated,isinRing]
        bond_features.append(bond_feat)
    g.ndata['feat'] = torch.FloatTensor(atom_features)
    g.edata['feat'] = torch.FloatTensor(bond_features)
    return g

def get_morgan_fingerprint(mol,radius=2,nbits=1024):##计算摩根指纹
    morgan_fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits), dtype=np.float32)
    return morgan_fp

class GraphData(Dataset):
    def __init__(self, strs, labels=None, train=True, add_hs=False, inchi=False):
        self.strs = strs##smiles or inchi
        self.train = train
        if self.train:
            self.labels = np.array(labels, dtype=np.float32)
            assert len(self.strs) == len(self.labels)
        self.add_hs = add_hs

        if inchi:
            self.read_mol_f = MolFromInchi
        else:
            self.read_mol_f = MolFromSmiles

    def __getitem__(self, idx):
        mol = self.read_mol_f(self.strs[idx])
        if self.add_hs:
            mol = AddHs(mol)
        if self.train:
            return (mol_to_graph(mol),get_morgan_fingerprint(mol),self.labels[idx, :])
        else:
            return (mol_to_graph(mol), get_morgan_fingerprint(mol))

    def __len__(self):
        return len(self.strs)

def collate_pair(samples):
    graphs_i, g_feats, labels = map(np.array, zip(*samples))
    batched_graph_i = dgl.batch(graphs_i)
    return (batched_graph_i,torch.as_tensor(g_feats),torch.as_tensor(labels))


class MPNNGNN(nn.Module):###图神经网络：graph → node embedding after MPNN message passing
    def __init__(self, node_in_feats, edge_in_feats, node_out_feats=64,
                 edge_hidden_feats=128, num_step_message_passing=6):
        super(MPNNGNN, self).__init__()

        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, node_out_feats),
            nn.ReLU()
        )
        self.num_step_message_passing = num_step_message_passing
        edge_network = nn.Sequential(
            nn.Linear(edge_in_feats, edge_hidden_feats),
            nn.ReLU(),
            nn.Linear(edge_hidden_feats, node_out_feats * node_out_feats)
        )
        self.gnn_layer = NNConv(
            in_feats=node_out_feats,
            out_feats=node_out_feats,
            edge_func=edge_network,
            aggregator_type='sum'
        )
        self.gru = nn.GRU(node_out_feats, node_out_feats)

    def forward(self, g, node_feats, edge_feats):
        node_feats = self.project_node_feats(node_feats) # (V, node_out_feats)
        hidden_feats = node_feats.unsqueeze(0)           # (1, V, node_out_feats)

        self.conv_acts,self.conv_grads = [],[]
        for _ in range(self.num_step_message_passing):
            ###hook
            # node_feats.register_hook(self.activations_hook)##eval时需要注释
            # self.conv_acts.append(node_feats)
            ###message passing
            node_feats = F.relu(self.gnn_layer(g, node_feats, edge_feats))
            node_feats, hidden_feats = self.gru(node_feats.unsqueeze(0), hidden_feats)
            node_feats = node_feats.squeeze(0)

        return node_feats

    # def activations_hook(self, grad):
    #     self.conv_grads.append(grad)

    # def get_intermediate_activations_gradients(self, output):
    #     output.backward(torch.ones_like(output))
    #     conv_grads = [conv_g.grad for conv_g in self.conv_grads]
    #     return self.conv_acts, self.conv_grads

class MPNNPredictor(nn.Module):##预测器：graph → prediction
    def __init__(self,node_in_feats,edge_in_feats,global_feats,node_out_feats=64,edge_hidden_feats=128,global_hidden_feats=512,
                 num_step_message_passing=6,num_step_set2set=6,num_layer_set2set=3):
        super(MPNNPredictor, self).__init__()

        self.gnn = MPNNGNN(node_in_feats=node_in_feats,node_out_feats=node_out_feats,edge_in_feats=edge_in_feats,
                           edge_hidden_feats=edge_hidden_feats,num_step_message_passing=num_step_message_passing,)
        self.readout = Set2Set(input_dim=node_out_feats,n_iters=num_step_set2set,n_layers=num_layer_set2set)

        self.global_subnet = nn.Sequential(nn.Linear(global_feats, global_hidden_feats),
                                           nn.ReLU(),
                                           nn.Linear(global_hidden_feats, global_hidden_feats),
                                           nn.ReLU())

        self.predict = nn.Sequential(nn.Linear(2 * node_out_feats + global_hidden_feats, node_out_feats),
                                     nn.ReLU(),
                                     nn.Linear(node_out_feats, 1))

    def forward(self, g, g_feat):
        n_fea,e_fea = g.ndata['feat'],g.edata['feat']
        g_fea = g_feat
        
        global_feats = self.global_subnet(g_fea)##摩根指纹处理
        node_feats = self.gnn(g, n_fea, e_fea)##MPNN执行信息传递
        graph_feats = self.readout(g, node_feats)##node embedding → graph embedding
        cat = torch.cat([graph_feats, global_feats], dim=1)##graph embedding+摩根指纹
        out = self.predict(cat)##全连接层预测
        return out

    def get_intermediate_activations_gradients(self, g, g_feat):
        n_fea,e_fea = g.ndata['feat'],g.edata['feat']
        g_fea = g_feat
        
        global_feats = self.global_subnet(g_fea)
        node_feats = self.gnn(g, n_fea, e_fea)
        graph_feats = self.readout(g, node_feats)
        cat = torch.cat([graph_feats, global_feats], dim=1)
        out = self.predict(cat)

        conv_acts, conv_grads = self.gnn.get_intermediate_activations_gradients(out)
        conv_grads = [conv for conv in conv_grads]
        return conv_acts, conv_grads

    def train_loop(self, loader, model, loss_fn, opt, DEVICE, progress=False):
        model = model.train()
        if progress:
            loader = tqdm(loader)
        loss_list = []
        for i,data in enumerate(loader):
            g, g_feat, label = data
            #print(i,':',g.num_nodes())
            g = g.to(DEVICE)
            g_feat = g_feat.to(DEVICE)
            label = label.to(DEVICE)
            opt.zero_grad()
            out = model(g, g_feat)
            loss = loss_fn(out, label)
            loss.backward()
            opt.step()
            loss_list.append(loss.item())
        return sum(loss_list)
    
    def eval_loop(self, loader, model, DEVICE, progress=False):
        model = model.eval()
        if progress:
            loader = tqdm(loader)
        y_exp_list,y_pred_list = [],[]
        with torch.no_grad():
            for g, g_feat, label in loader:
                g = g.to(DEVICE)
                g_feat = g_feat.to(DEVICE)
                out = model(g, g_feat)
                y_exp_list.append(label.cpu())
                y_pred_list.append(out.cpu())
        return torch.cat(y_exp_list), torch.cat(y_pred_list)