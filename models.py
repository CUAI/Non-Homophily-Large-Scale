import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import GCNConv, SGConv, GATConv, JumpingKnowledge, APPNP, GCN2Conv, MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import numpy as np
import scipy.sparse
from tqdm import tqdm


class LINK(nn.Module):
    """ logistic regression on adjacency matrix """
    
    def __init__(self, num_nodes, out_channels):
        super(LINK, self).__init__()
        self.W = nn.Linear(num_nodes, out_channels)
        self.num_nodes = num_nodes

    def reset_parameters(self):
        self.W.reset_parameters()
        
    def forward(self, data):
        N = data.graph['num_nodes']
        edge_index = data.graph['edge_index']
        if isinstance(edge_index, torch.Tensor):
            row, col = edge_index
            row = row-row.min() # for sampling
            A = SparseTensor(row=row, col=col, sparse_sizes=(N, self.num_nodes)).to_torch_sparse_coo_tensor()
        elif isinstance(edge_index, SparseTensor):
            A = edge_index.to_torch_sparse_coo_tensor()
        logits = self.W(A)
        return logits


class MLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data, input_tensor=False):
        if not input_tensor:
            x = data.graph['node_feat']
        else:
            x = data
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x
    
class SGC(nn.Module):
    def __init__(self, in_channels, out_channels, hops):
        """ takes 'hops' power of the normalized adjacency"""
        super(SGC, self).__init__()
        self.conv = SGConv(in_channels, out_channels, hops, cached=True) 

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, data):
        edge_index = data.graph['edge_index']
        x = data.graph['node_feat']
        x = self.conv(x, edge_index)
        return x


class SGCMem(nn.Module):
    def __init__(self, in_channels, out_channels, hops):
        """ lower memory version (if out_channels < in_channels)
        takes weight multiplication first, then propagate
        takes hops power of the normalized adjacency
        """
        super(SGCMem, self).__init__()

        self.lin = nn.Linear(in_channels, out_channels)
        self.hops = hops

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, data):
        edge_index = data.graph['edge_index']
        x = data.graph['node_feat']
        x = self.lin(x)
        n = data.graph['num_nodes']
        edge_weight=None

        if isinstance(edge_index, torch.Tensor):
            edge_index, edge_weight = gcn_norm( 
                edge_index, edge_weight, n, False,
                 dtype=x.dtype)
            row, col = edge_index
            adj_t = SparseTensor(row=col, col=row, value=edge_weight, sparse_sizes=(n, n))
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, n, False,
                dtype=x.dtype)
            edge_weight=None
            adj_t = edge_index

        for _ in range(self.hops):
            x = matmul(adj_t, x)
        
        return x


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, save_mem=False, use_bn=True):
        super(GCN, self).__init__()

        cached = False
        add_self_loops = True
        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(
            GCNConv(hidden_channels, out_channels, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()


    def forward(self, data):
        x = data.graph['node_feat']
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, data.graph['edge_index'])
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, data.graph['edge_index'])
        return x


class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, heads=2, sampling=False, add_self_loops=True):
        super(GAT, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=heads, concat=True, add_self_loops=add_self_loops))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels*heads))
        for _ in range(num_layers - 2):

            self.convs.append(
                    GATConv(hidden_channels*heads, hidden_channels, heads=heads, concat=True, add_self_loops=add_self_loops) ) 
            self.bns.append(nn.BatchNorm1d(hidden_channels*heads))

        self.convs.append(
            GATConv(hidden_channels*heads, out_channels, heads=heads, concat=False, add_self_loops=add_self_loops))

        self.dropout = dropout
        self.activation = F.elu 
        self.sampling = sampling
        self.num_layers = num_layers

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()


    def forward(self, data, adjs=None, x_batch=None):
        if not self.sampling:
            x = data.graph['node_feat']
            for i, conv in enumerate(self.convs[:-1]):
                x = conv(x, data.graph['edge_index'])
                x = self.bns[i](x)
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[-1](x, data.graph['edge_index'])
        else:
            x = x_batch
            for i, (edge_index, _, size) in enumerate(adjs):
                x_target = x[:size[1]]  # Target nodes are always placed first.
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = self.bns[i](x)
                    x = self.activation(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
        return x


    def inference(self, data, subgraph_loader):
        x_all = data.graph['node_feat'] 
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        total_edges = 0
        device = x_all.device
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = self.bns[i](x)
                    x = self.activation(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all

class MultiLP(nn.Module):
    """ label propagation, with possibly multiple hops of the adjacency """
    
    def __init__(self, out_channels, alpha, hops, num_iters=50, mult_bin=False):
        super(MultiLP, self).__init__()
        self.out_channels = out_channels
        self.alpha = alpha
        self.hops = hops
        self.num_iters = num_iters
        self.mult_bin = mult_bin # handle multiple binary tasks
        
    def forward(self, data, train_idx):
        n = data.graph['num_nodes']
        edge_index = data.graph['edge_index']
        edge_weight=None

        if isinstance(edge_index, torch.Tensor):
            edge_index, edge_weight = gcn_norm( 
                edge_index, edge_weight, n, False)
            row, col = edge_index
            # transposed if directed
            adj_t = SparseTensor(row=col, col=row, value=edge_weight, sparse_sizes=(n, n))
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, n, False)
            edge_weight=None
            adj_t = edge_index

        y = torch.zeros((n, self.out_channels)).to(adj_t.device())
        if data.label.shape[1] == 1:
            # make one hot
            y[train_idx] = F.one_hot(data.label[train_idx], self.out_channels).squeeze(1).to(y)
        elif self.mult_bin:
            y = torch.zeros((n, 2*self.out_channels)).to(adj_t.device())
            for task in range(data.label.shape[1]):
                y[train_idx, 2*task:2*task+2] = F.one_hot(data.label[train_idx, task], 2).to(y)
        else:
            y[train_idx] = data.label[train_idx].to(y.dtype)
        result = y.clone()
        for _ in range(self.num_iters):
            for _ in range(self.hops):
                result = matmul(adj_t, result)
            result *= self.alpha
            result += (1-self.alpha)*y

        if self.mult_bin:
            output = torch.zeros((n, self.out_channels)).to(result.device)
            for task in range(data.label.shape[1]):
                output[:, task] = result[:, 2*task+1]
            result = output

        return result


class MixHopLayer(nn.Module):
    """ Our MixHop layer """
    def __init__(self, in_channels, out_channels, hops=2):
        super(MixHopLayer, self).__init__()
        self.hops = hops
        self.lins = nn.ModuleList()
        for hop in range(self.hops+1):
            lin = nn.Linear(in_channels, out_channels)
            self.lins.append(lin)

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, adj_t):
        xs = [self.lins[0](x) ]
        for j in range(1,self.hops+1):
            # less runtime efficient but usually more memory efficient to mult weight matrix first
            x_j = self.lins[j](x)
            for hop in range(j):
                x_j = matmul(adj_t, x_j)
            xs += [x_j]
        return torch.cat(xs, dim=1)

class MixHop(nn.Module):
    """ our implementation of MixHop
    some assumptions: the powers of the adjacency are [0, 1, ..., hops],
        with every power in between
    each concatenated layer has the same dimension --- hidden_channels
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, hops=2):
        super(MixHop, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(MixHopLayer(in_channels, hidden_channels, hops=hops))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels*(hops+1)))
        for _ in range(num_layers - 2):
            self.convs.append(
                MixHopLayer(hidden_channels*(hops+1), hidden_channels, hops=hops))
            self.bns.append(nn.BatchNorm1d(hidden_channels*(hops+1)))

        self.convs.append(
            MixHopLayer(hidden_channels*(hops+1), out_channels, hops=hops))

        # note: uses linear projection instead of paper's attention output
        self.final_project = nn.Linear(out_channels*(hops+1), out_channels)

        self.dropout = dropout
        self.activation = F.relu

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.final_project.reset_parameters()


    def forward(self, data):
        x = data.graph['node_feat']
        n = data.graph['num_nodes']
        edge_index = data.graph['edge_index']
        edge_weight = None
        if isinstance(edge_index, torch.Tensor):
            edge_index, edge_weight = gcn_norm( 
                edge_index, edge_weight, n, False,
                 dtype=x.dtype)
            row, col = edge_index
            adj_t = SparseTensor(row=col, col=row, value=edge_weight, sparse_sizes=(n, n))
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, n, False,
                dtype=x.dtype)
            edge_weight=None
            adj_t = edge_index
        
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)

        x = self.final_project(x)
        return x

class GCNJK(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, save_mem=False, jk_type='max'):
        super(GCNJK, self).__init__()

        cached = False
        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=cached, normalize=not save_mem))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=cached, normalize=not save_mem))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(
            GCNConv(hidden_channels, hidden_channels, cached=cached, normalize=not save_mem))

        self.dropout = dropout
        self.activation = F.relu
        self.jump = JumpingKnowledge(jk_type, channels=hidden_channels, num_layers=1)
        if jk_type == 'cat':
            self.final_project = nn.Linear(hidden_channels * num_layers, out_channels)
        else: # max or lstm
            self.final_project = nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.jump.reset_parameters()
        self.final_project.reset_parameters()

    def forward(self, data):
        x = data.graph['node_feat']
        xs = []
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, data.graph['edge_index'])
            x = self.bns[i](x)
            x = self.activation(x)
            xs.append(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, data.graph['edge_index'])
        xs.append(x)
        x = self.jump(xs)
        x = self.final_project(x)
        return x

class GATJK(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, heads=2, jk_type='max'):
        super(GATJK, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=heads, concat=True))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels*heads))
        for _ in range(num_layers - 2):

            self.convs.append(
                    GATConv(hidden_channels*heads, hidden_channels, heads=heads, concat=True) ) 
            self.bns.append(nn.BatchNorm1d(hidden_channels*heads))

        self.convs.append(
            GATConv(hidden_channels*heads, hidden_channels, heads=heads))

        self.dropout = dropout
        self.activation = F.elu # note: uses elu

        self.jump = JumpingKnowledge(jk_type, channels=hidden_channels*heads, num_layers=1)
        if jk_type == 'cat':
            self.final_project = nn.Linear(hidden_channels*heads*num_layers, out_channels)
        else: # max or lstm
            self.final_project = nn.Linear(hidden_channels*heads, out_channels)


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.jump.reset_parameters()
        self.final_project.reset_parameters()


    def forward(self, data):
        x = data.graph['node_feat']
        xs = []
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, data.graph['edge_index'])
            x = self.bns[i](x)
            x = self.activation(x)
            xs.append(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, data.graph['edge_index'])
        xs.append(x)
        x = self.jump(xs)
        x = self.final_project(x)
        return x

class H2GCNConv(nn.Module):
    """ Neighborhood aggregation step """
    def __init__(self):
        super(H2GCNConv, self).__init__()

    def reset_parameters(self):
        pass

    def forward(self, x, adj_t, adj_t2):
        x1 = matmul(adj_t, x)
        x2 = matmul(adj_t2, x)
        return torch.cat([x1, x2], dim=1)


class H2GCN(nn.Module):
    """ our implementation """
    def __init__(self, in_channels, hidden_channels, out_channels, edge_index, num_nodes,
                    num_layers=2, dropout=0.5, save_mem=False, num_mlp_layers=1,
                    use_bn=True, conv_dropout=True):
        super(H2GCN, self).__init__()

        self.feature_embed = MLP(in_channels, hidden_channels,
                hidden_channels, num_layers=num_mlp_layers, dropout=dropout)


        self.convs = nn.ModuleList()
        self.convs.append(H2GCNConv())

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels*2*len(self.convs) ) )

        for l in range(num_layers - 1):
            self.convs.append(H2GCNConv())
            if l != num_layers-2:
                self.bns.append(nn.BatchNorm1d(hidden_channels*2*len(self.convs) ) )

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.conv_dropout = conv_dropout # dropout neighborhood aggregation steps

        self.jump = JumpingKnowledge('cat')
        last_dim = hidden_channels*(2**(num_layers+1)-1)
        self.final_project = nn.Linear(last_dim, out_channels)

        self.num_nodes = num_nodes
        self.init_adj(edge_index)

    def reset_parameters(self):
        self.feature_embed.reset_parameters()
        self.final_project.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def init_adj(self, edge_index):
        """ cache normalized adjacency and normalized strict two-hop adjacency,
        neither has self loops
        """
        n = self.num_nodes
        
        if isinstance(edge_index, SparseTensor):
            dev = edge_index.device
            adj_t = edge_index
            adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
            adj_t[adj_t > 0] = 1
            adj_t[adj_t < 0] = 0
            adj_t = SparseTensor.from_scipy(adj_t).to(dev)
        elif isinstance(edge_index, torch.Tensor):
            row, col = edge_index
            adj_t = SparseTensor(row=col, col=row, value=None, sparse_sizes=(n, n))

        adj_t.remove_diag(0)
        adj_t2 = matmul(adj_t, adj_t)
        adj_t2.remove_diag(0)
        adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
        adj_t2 = scipy.sparse.csr_matrix(adj_t2.to_scipy())
        adj_t2 = adj_t2 - adj_t
        adj_t2[adj_t2 > 0] = 1
        adj_t2[adj_t2 < 0] = 0

        adj_t = SparseTensor.from_scipy(adj_t)
        adj_t2 = SparseTensor.from_scipy(adj_t2)
        
        adj_t = gcn_norm(adj_t, None, n, add_self_loops=False)
        adj_t2 = gcn_norm(adj_t2, None, n, add_self_loops=False)

        self.adj_t = adj_t.to(edge_index.device)
        self.adj_t2 = adj_t2.to(edge_index.device)



    def forward(self, data):
        x = data.graph['node_feat']
        n = data.graph['num_nodes']

        adj_t = self.adj_t
        adj_t2 = self.adj_t2
        
        x = self.feature_embed(data)
        x = self.activation(x)
        xs = [x]
        if self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t, adj_t2) 
            if self.use_bn:
                x = self.bns[i](x)
            xs.append(x)
            if self.conv_dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t, adj_t2)
        if self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        xs.append(x)

        x = self.jump(xs)
        if not self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.final_project(x)
        return x


class APPNP_Net(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dprate=.0, dropout=.5, K=10, alpha=.1, num_layers=3):
        super(APPNP_Net, self).__init__()
        
        self.mlp = MLP(in_channels, hidden_channels, out_channels, num_layers=num_layers, dropout=dropout)
        self.prop1 = APPNP(K, alpha)

        self.dprate = dprate
        self.dropout = dropout
        
    def reset_parameters(self):
        self.mlp.reset_parameters()
        self.prop1.reset_parameters()

    def forward(self, data):
        edge_index = data.graph['edge_index']
        x = self.mlp(data)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return x
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return x


class LINK_Concat(nn.Module):	
    """ concate A and X as joint embeddings i.e. MLP([A;X])"""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_nodes, dropout=.5, cache=True):	
        super(LINK_Concat, self).__init__()	
        self.mlp = MLP(in_channels + num_nodes, hidden_channels, out_channels, num_layers, dropout=dropout)
        self.in_channels = in_channels	
        self.cache = cache
        self.x = None

    def reset_parameters(self):	
        self.mlp.reset_parameters()	

    def forward(self, data):	
        if (not self.cache) or (not isinstance(self.x, torch.Tensor)):
                N = data.graph['num_nodes']	
                feat_dim = data.graph['node_feat']	
                row, col = data.graph['edge_index']	
                col = col + self.in_channels	
                feat_nz = data.graph['node_feat'].nonzero(as_tuple=True)	
                feat_row, feat_col = feat_nz	
                full_row = torch.cat((feat_row, row))	
                full_col = torch.cat((feat_col, col))	
                value = data.graph['node_feat'][feat_nz]	
                full_value = torch.cat((value, 	
                                torch.ones(row.shape[0], device=value.device)))	
                x = SparseTensor(row=full_row, col=full_col,	
                         sparse_sizes=(N, N+self.in_channels)	
                            ).to_torch_sparse_coo_tensor()	
                if self.cache:
                    self.x = x
        else:
                x = self.x
        logits = self.mlp(x, input_tensor=True)
        return logits


class LINKX(nn.Module):	
    """ our LINKX method with skip connections 
        a = MLP_1(A), x = MLP_2(X), MLP_3(sigma(W_1[a, x] + a + x))
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_nodes, dropout=.5, cache=False, inner_activation=False, inner_dropout=False, init_layers_A=1, init_layers_X=1):
        super(LINKX, self).__init__()	
        self.mlpA = MLP(num_nodes, hidden_channels, hidden_channels, init_layers_A, dropout=0)
        self.mlpX = MLP(in_channels, hidden_channels, hidden_channels, init_layers_X, dropout=0)
        self.W = nn.Linear(2*hidden_channels, hidden_channels)
        self.mlp_final = MLP(hidden_channels, hidden_channels, out_channels, num_layers, dropout=dropout)
        self.in_channels = in_channels
        self.num_nodes = num_nodes
        self.A = None
        self.inner_activation = inner_activation
        self.inner_dropout = inner_dropout

    def reset_parameters(self):	
        self.mlpA.reset_parameters()	
        self.mlpX.reset_parameters()
        self.W.reset_parameters()
        self.mlp_final.reset_parameters()	

    def forward(self, data):	
        m = data.graph['num_nodes']	
        feat_dim = data.graph['node_feat']	
        row, col = data.graph['edge_index']	
        row = row-row.min()
        A = SparseTensor(row=row, col=col,	
                 sparse_sizes=(m, self.num_nodes)
                        ).to_torch_sparse_coo_tensor()

        xA = self.mlpA(A, input_tensor=True)
        xX = self.mlpX(data.graph['node_feat'], input_tensor=True)
        x = torch.cat((xA, xX), axis=-1)
        x = self.W(x)
        if self.inner_dropout:
            x = F.dropout(x)
        if self.inner_activation:
            x = F.relu(x)
        x = F.relu(x + xA + xX)
        x = self.mlp_final(x, input_tensor=True)

        return x

class GPR_prop(MessagePassing):
    '''
    GPRGNN, from original repo https://github.com/jianhao2016/GPRGNN
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like
            TEMP = 0.0*np.ones(K+1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = nn.Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K

    def forward(self, x, edge_index, edge_weight=None):
        if isinstance(edge_index, torch.Tensor):
            edge_index, norm = gcn_norm(
                edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
            norm = None

        hidden = x*(self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k+1]
            hidden = hidden + gamma*x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)

class GPRGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, Init='Random', dprate=.0, dropout=.5, K=10, alpha=.1, Gamma=None, num_layers=3):
        super(GPRGNN, self).__init__()
        
        self.mlp = MLP(in_channels, hidden_channels, out_channels, num_layers=num_layers, dropout=dropout)
        self.prop1 = GPR_prop(K, alpha, Init, Gamma)

        self.Init = Init
        self.dprate = dprate
        self.dropout = dropout
        
    def reset_parameters(self):
        self.mlp.reset_parameters()
        self.prop1.reset_parameters()

    def forward(self, data):
        edge_index = data.graph['edge_index']
        x = self.mlp(data)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return x
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return x


class GCNII(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, alpha, theta, shared_weights=True, dropout=0.5):
        super(GCNII, self).__init__()

        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))
        
        self.bns = nn.ModuleList()
        self.convs = nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=False))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        self.dropout = dropout
        
    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x = data.graph['node_feat']
        n = data.graph['num_nodes']
        edge_index = data.graph['edge_index']
        edge_weight = None
        if isinstance(edge_index, torch.Tensor):
            edge_index, edge_weight = gcn_norm( 
                edge_index, edge_weight, n, False, dtype=x.dtype)
            row, col = edge_index
            adj_t = SparseTensor(row=col, col=row, value=edge_weight, sparse_sizes=(n, n))
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, n, False, dtype=x.dtype)
            edge_weight=None
            adj_t = edge_index
        
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()
        
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, adj_t)
            x = self.bns[i](x)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)
        return x
