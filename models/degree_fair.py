import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sp
import torch.optim as optim
from layers import *




def convert_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class DFairGNN_2(nn.Module):
    def __init__(self, args, nfeat, nclass, max_degree, sam):
        super(DFairGNN_2, self).__init__()
        self.dropout = args.dropout
        self.debias1 = Debias_v2(args, nfeat, args.dim, max_degree, sam)
        self.debias2 = Debias_v2(args, args.dim, args.dim, max_degree, sam)
        self.fc = nn.Linear(args.dim, nclass)


    def forward(self, x, adj, d, idx):
        x, b1, film1 = self.debias1(x, adj, d, idx)
        x = F.dropout(x, self.dropout, training=self.training)

        x, b2, film2 = self.debias2(x, adj, d, idx)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc(x)

        return x, b1+b2, film1+film2




class DFair_GCN(nn.Module):
    def __init__(self, args, nfeat, nclass, max_degree):
        super(DFair_GCN, self).__init__()
        self.dropout = args.dropout
        self.debias1 = Debias_v2(args, nfeat, args.dim, max_degree)
        self.debias2 = Debias_v2(args, args.dim, args.dim, max_degree)
        self.fc = nn.Linear(args.dim, nclass)


    def forward(self, x, adj, d, idx, edge):
        x, b1, film1 = self.debias1(x, adj, d, idx, edge)
        x = F.dropout(x, self.dropout, training=self.training)

        x, b2, film2 = self.debias2(x, adj, d, idx, edge)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc(x)

        return F.log_softmax(x, dim=1), b1+b2, film1+film2


class DFair_GAT(nn.Module):
    def __init__(self, args, nfeat, nclass, max_degree, nhid=16, nheads=3):
        super(DFair_GAT, self).__init__()
        self.dropout = args.dropout

        self.attentions = [Debias_v2(args, nfeat, nhid, max_degree) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = Debias_v2(args, nhid*nheads, nclass, max_degree)


    def forward(self, x, adj, d, idx, edge):
        
        x = F.dropout(x, self.dropout, training=self.training)
        
        x1 = []
        b1 = 0
        film1 = 0
        for att in self.attentions:
            at_x1, at_b1, at_film1 = att(x, adj, d, idx, edge) 
            x1.append(at_x1)
            b1 += at_b1
            film1 += at_film1

        x = torch.cat(x1, dim=1)
        b1 /= len(self.attentions)
        film1 /= len(self.attentions)

        x = F.dropout(x, self.dropout, training=self.training)
        x, b2, film2 = self.out_att(x, adj, d, idx, edge)

        return F.log_softmax(x, dim=1), b1+b2, film1+film2


class DFair_Sage(nn.Module):
    def __init__(self, args, nfeat, nclass, max_degree, nhid1=16, nhid2=8):
        super(DFair_Sage, self).__init__()
        self.dropout = args.dropout
        self.debias1 = Debias_v2(args, nfeat, nhid1, max_degree)
        self.debias2 = Debias_v2(args, nhid1*2, nhid2, max_degree)
        self.fc = nn.Linear(nhid2*2, nclass)


    def forward(self, x, adj, d, idx, edge):

        #x = F.dropout(x, self.dropout, training=self.training)
        x, b1, film1 = self.debias1(x, adj, d, idx, edge)
        x = F.dropout(x, self.dropout, training=self.training)

        x, b2, film2 = self.debias2(x, adj, d, idx, edge)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc(x)

        return F.log_softmax(x, dim=1), b1+b2, film1+film2
