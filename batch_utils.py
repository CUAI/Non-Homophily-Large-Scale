import argparse
import sys
import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, sort_edge_index
from torch_geometric.data import NeighborSampler, ClusterData, ClusterLoader, Data, GraphSAINTNodeSampler, GraphSAINTEdgeSampler, GraphSAINTRandomWalkSampler, RandomNodeSampler
from torch_scatter import scatter

from logger import Logger, SimpleLogger
from dataset import load_nc_dataset, NCDataset
from data_utils import normalize, gen_normalized_adjs, evaluate, eval_acc, eval_rocauc, to_sparse_tensor
from parse import parse_method, parser_add_main_args


def nc_dataset_to_torch_geo(dataset, idx, device=torch.device('cpu')):
    tg_data = Data()
    tg_data.x = dataset.graph['node_feat']
    tg_data.edge_index = dataset.graph['edge_index']
    tg_data.edge_attr = dataset.graph['edge_feat']
    tg_data.y = dataset.label
    mask = torch.zeros(tg_data.num_nodes, dtype=torch.bool, device=device)
    mask[idx] = True
    tg_data.node_ids = torch.arange(tg_data.num_nodes, device=device)
    tg_data.mask = mask
    return tg_data

def torch_geo_to_nc_dataset(tg_data, name='', device=torch.device('cpu')):
    dataset = NCDataset(name)
    dataset.label = tg_data.y.to(device)
    dataset.graph['node_feat'] = tg_data.x.to(device)
    dataset.graph['edge_index'] = tg_data.edge_index.to(device)
    dataset.graph['edge_feat'] = tg_data.edge_attr
    dataset.graph['num_nodes'] = dataset.graph['node_feat'].shape[0]
    return dataset

    
class AdjRowLoader():
    def __init__(self, dataset, idx, num_parts=100, full_epoch=False):
        """
        if not full_epoch, then just return one chunk of nodes
        """
        self.dataset = dataset
        self.full_epoch = full_epoch
        n = dataset.graph['num_nodes']
        self.node_feat = dataset.graph['node_feat']
        self.edge_index = dataset.graph['edge_index']
        self.edge_index = sort_edge_index(self.edge_index)[0]
        self.part_spots = [0]
        self.part_nodes = [0]
        self.idx = idx
        self.mask = torch.zeros(dataset.graph['num_nodes'], dtype=torch.bool)#, device=device)
        self.mask[idx] = True
        num_edges = self.edge_index.shape[1]
        approx_size = num_edges // num_parts
        approx_part_spots = list(range(approx_size, num_edges, approx_size))[:-1]
        for idx in approx_part_spots:
            curr_node = self.edge_index[0,idx].item()
            curr_idx = idx
            while curr_idx < self.edge_index.shape[1] and self.edge_index[0,curr_idx] == curr_node:
                curr_idx += 1
            self.part_nodes.append(self.edge_index[0, curr_idx].item())
            self.part_spots.append(curr_idx)
        self.part_nodes.append(n)
        self.part_spots.append(self.edge_index.shape[1])
    
    def __iter__(self):
        self.k = 0
        return self
    
    def __next__(self):
        if self.k >= len(self.part_spots)-1:
            raise StopIteration
            
        if not self.full_epoch:
            self.k = np.random.randint(len(self.part_spots)-1)
            
        tg_data = Data()
        batch_edge_index = self.edge_index[:, self.part_spots[self.k]:self.part_spots[self.k+1]]
        node_ids = list(range(self.part_nodes[self.k], self.part_nodes[self.k+1]))
        tg_data.node_ids = node_ids
        tg_data.edge_index = batch_edge_index
        batch_node_feat = self.node_feat[node_ids]
        tg_data.x = batch_node_feat
        tg_data.edge_attr = None
        tg_data.y = self.dataset.label[node_ids]
        tg_data.num_nodes = len(node_ids)
        mask = self.mask[node_ids]
        tg_data.mask = mask
        self.k += 1
        
        if not self.full_epoch:
            self.k = float('inf')
        return tg_data
    

def make_loader(args, dataset, idx, mini_batch=True, device=torch.device('cpu'), test=False):
    if not mini_batch:
        tg_data = nc_dataset_to_torch_geo(dataset, idx, device=device)
        # full batch test right now
        loader = RandomNodeSampler(tg_data, num_parts=1, shuffle=True, num_workers=0)
        return loader
        
    if args.train_batch == 'cluster':
        tg_data = nc_dataset_to_torch_geo(dataset, idx, device=device)
        cluster_data = ClusterData(tg_data, num_parts=args.num_parts)
        loader = ClusterLoader(cluster_data, batch_size=args.cluster_batch_size, shuffle=True, num_workers=0)
        
    elif args.train_batch == 'graphsaint-node':
        tg_data = nc_dataset_to_torch_geo(dataset, idx, device=device)
        if not test:
            loader = GraphSAINTNodeSampler(tg_data, batch_size=args.batch_size, shuffle=True, num_workers=0, num_steps=args.saint_num_steps)
        else:
            loader = RandomNodeSampler(tg_data, num_parts=args.test_num_parts, shuffle=True, num_workers=0)
            
    elif args.train_batch == 'graphsaint-edge':
        tg_data = nc_dataset_to_torch_geo(dataset, idx, device=device)
        if not test:
            loader = GraphSAINTEdgeSampler(tg_data, batch_size=args.batch_size, shuffle=True, num_workers=0, num_steps=args.saint_num_steps)
        else:
            loader = RandomNodeSampler(tg_data, num_parts=args.test_num_parts, shuffle=True, num_workers=0)
            
    elif args.train_batch == 'graphsaint-rw':
        tg_data = nc_dataset_to_torch_geo(dataset, idx, device=device)
        if not test:
            loader = GraphSAINTRandomWalkSampler(tg_data, batch_size=args.batch_size, walk_length=args.num_layers, shuffle=True, num_workers=0, num_steps=args.saint_num_steps)
        else:
            loader = RandomNodeSampler(tg_data, num_parts=args.test_num_parts, shuffle=True, num_workers=0)
            
    elif args.train_batch == 'random':
        tg_data = nc_dataset_to_torch_geo(dataset, idx, device=device)
        loader = RandomNodeSampler(tg_data, num_parts=args.num_parts, shuffle=True, num_workers=0)
        
    elif args.train_batch == 'full-batch':
        tg_data = nc_dataset_to_torch_geo(dataset, idx, device=device)
        loader = RandomNodeSampler(tg_data, num_parts=1, shuffle=True, num_workers=0)
        
    elif args.train_batch == 'row':
        loader = AdjRowLoader(dataset, idx, num_parts=args.num_parts, full_epoch=test)
        
    else:
        raise ValueError('Invalid train batching')
        
    return loader