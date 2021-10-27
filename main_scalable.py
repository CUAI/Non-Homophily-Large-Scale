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
from batch_utils import nc_dataset_to_torch_geo, torch_geo_to_nc_dataset, AdjRowLoader, make_loader


# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
np.random.seed(0)

### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
parser.add_argument('--train_batch', type=str, default='cluster', help='type of mini batch loading scheme for training GNN')
parser.add_argument('--no_mini_batch_test', action='store_true', help='whether to test on mini batches as well')
parser.add_argument('--batch_size', type=int, default=10000)
parser.add_argument('--num_parts', type=int, default=100, help='number of partitions for partition batching')
parser.add_argument('--cluster_batch_size', type=int, default=1, help='number of clusters to use per cluster-gcn step')
parser.add_argument('--saint_num_steps', type=int, default=5, help='number of steps for graphsaint')
parser.add_argument('--test_num_parts', type=int, default=10, help='number of partitions for testing')
args = parser.parse_args()
print(args)

device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
if args.cpu:
    device = torch.device('cpu')

### Load and preprocess data ###
dataset = load_nc_dataset(args.dataset, args.sub_dataset)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)
dataset.label = dataset.label.to(device)

split_idx = dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
train_idx = split_idx['train']
train_idx = train_idx.to(device)

n = dataset.graph['num_nodes']
# infer the number of classes for non one-hot and one-hot labels
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

# whether or not to symmetrize matters a lot!! pay attention to this
# e.g. directed edges are temporally useful in arxiv-year,
# so we usually do not symmetrize, but for label prop symmetrizing helps
if not args.directed and args.dataset != 'ogbn-proteins':
    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])

train_loader, subgraph_loader = None, None

print(f"num nodes {n} | num classes {c} | num node feats {d}")

### Load method ###

model = parse_method(args, dataset, n, c, d, device)


# using rocauc as the eval function
if args.rocauc or args.dataset in ('yelp-chi', 'twitch-e', 'ogbn-proteins'):
    criterion = nn.BCEWithLogitsLoss()
    eval_func = eval_rocauc
else:
    criterion = nn.NLLLoss()
    eval_func = eval_acc

logger = Logger(args.runs, args)

model.train()
print('MODEL:', model)



def train():
    model.train()

    total_loss = 0
    for tg_batch in train_loader:
        batch_train_idx = tg_batch.mask.to(torch.bool)
        batch_dataset = torch_geo_to_nc_dataset(tg_batch, device=device)
        optimizer.zero_grad()
        out = model(batch_dataset)
        if args.rocauc or args.dataset in ('yelp-chi', 'twitch-e', 'ogbn-proteins'):
            if dataset.label.shape[1] == 1:
                # change -1 instances to 0 for one-hot transform
                # dataset.label[dataset.label==-1] = 0
                true_label = F.one_hot(batch_dataset.label, batch_dataset.label.max() + 1).squeeze(1)
            else:
                true_label = batch_dataset.label

            loss = criterion(out[batch_train_idx], true_label[batch_train_idx].to(out.dtype))
        else:
            out = F.log_softmax(out, dim=1)
            loss = criterion(
                out[batch_train_idx], batch_dataset.label.squeeze(1)[batch_train_idx])
        total_loss += loss
        loss.backward()
        optimizer.step()
 
    return total_loss

def test():
    # needs a loader that includes every node in the graph
    model.eval()
    
    full_out = torch.zeros(n, c, device=device)
    with torch.no_grad():
        for tg_batch in test_loader:
            node_ids = tg_batch.node_ids
            batch_dataset = torch_geo_to_nc_dataset(tg_batch, device=device)
            out = model(batch_dataset)
            full_out[node_ids] = out
    result = evaluate(model, dataset, split_idx, eval_func, result=full_out, sampling=args.sampling, subgraph_loader=subgraph_loader)
    logger.add_result(run, result[:-1])
    return result


### Training loop ###
for run in range(args.runs):
    train_idx = split_idx['train']
    train_idx = train_idx.to(device)

    print('making train loader')
    train_loader = make_loader(args, dataset, train_idx, device=device)
    if not args.no_mini_batch_test:
        test_loader = make_loader(args, dataset, train_idx, device=device, test=True)
    else:
        test_loader = make_loader(args, dataset, split_idx['test'], mini_batch = False, device=device)

    model.reset_parameters()
    if args.adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = float('-inf')
    for epoch in range(args.epochs):
        total_loss = train()
        result = test()

        if result[1] > best_val:
            best_out = F.log_softmax(result[-1], dim=1)
            best_val = result[1]

        if epoch % args.display_step == 0:
            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {total_loss:.4f}, '
                  f'Train: {100 * result[0]:.2f}%, '
                  f'Valid: {100 * result[1]:.2f}%, '
                  f'Test: {100 * result[2]:.2f}%')
            if args.print_prop:
                pred = out.argmax(dim=-1, keepdim=True)
                print("Predicted proportions:", pred.unique(return_counts=True)[1].float()/pred.shape[0])
    logger.print_statistics(run)

    split_idx = dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)


### Save results ###
best_val, best_test = logger.print_statistics()
filename = f'results/{args.dataset}.csv'
print(f"Saving results to {filename}")
with open(f"{filename}", 'a+') as write_obj:
    sub_dataset = f'{args.sub_dataset},' if args.sub_dataset else ''
    write_obj.write(f"{args.method}," + f"{sub_dataset}" +
                    f"{best_val.mean():.3f} ± {best_val.std():.3f}," +
                    f"{best_test.mean():.3f} ± {best_test.std():.3f}\n")
