import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math
import scipy.sparse as sp
import numpy as np


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class Debias_v2(nn.Module):
    def __init__(self, args, in_feat, out_feat, d_max):
        super(Debias_v2, self).__init__()

        self.dim_M = args.dim_d
        self.out_feat = out_feat
        self.omega = args.omega
        self.d_max = (d_max+1) #0->dmax
        self.base = args.base
        self.dataset = args.dataset
        self.k = args.k
        #self.w = args.w
        #self.sparse = args.sparse

        
        self.weight = nn.Linear(in_feat, out_feat)
        if args.base == 2:
            self.a = nn.Parameter(torch.zeros(size=(1, 2*out_feat)))
            nn.init.xavier_uniform_(self.a.data, gain=1.414)
            self.special_spmm = SpecialSpmm()
            self.dropout = nn.Dropout()

        self.W_gamma = nn.Parameter(torch.FloatTensor(self.dim_M, out_feat))
        self.W_beta = nn.Parameter(torch.FloatTensor(self.dim_M, out_feat))
        self.U_gamma = nn.Parameter(torch.FloatTensor(out_feat, out_feat))
        self.U_beta = nn.Parameter(torch.FloatTensor(out_feat, out_feat))
        self.b_gamma = nn.Parameter(torch.FloatTensor(1, out_feat))
        self.b_beta = nn.Parameter(torch.FloatTensor(1, out_feat))

        self.W_add = nn.Linear(out_feat, out_feat, bias=False)
        self.W_rev = nn.Linear(out_feat, out_feat, bias=False)

        # Positional Encoding
        PE = np.array([
            [pos / np.power(10000, (i-i%2)/self.dim_M) for i in range(self.dim_M)]
            for pos in range(self.d_max)])

        PE[:, 0::2] = np.sin(PE[:, 0::2]) 
        PE[:, 1::2] = np.cos(PE[:, 1::2]) 
        self.PE = torch.as_tensor(PE, dtype=torch.float32)
        
        self.set_parameters()


    def set_parameters(self):
        #nn.init.uniform_(self.m)
        nn.init.uniform_(self.W_gamma)
        nn.init.uniform_(self.W_beta)
        nn.init.uniform_(self.U_gamma)
        nn.init.uniform_(self.U_beta)
        nn.init.uniform_(self.b_gamma)
        nn.init.uniform_(self.b_beta)

        '''
            M_stdv = 1. / math.sqrt(self.M.size(1))
            self.M.data.uniform_(-M_stdv, M_stdv)

            b_stdv = 1. / math.sqrt(self.b.size(1))
            self.b.data.uniform_(-b_stdv, b_stdv)

            for m in self.modules():
                print(m.weight)
        '''

    def forward(self, x, adj, degree, idx, edge):
        h = self.weight(x)
        
        m_dv = torch.squeeze(self.PE[degree])
        m_dv = m_dv.cuda()

        # version 1
        if self.dataset != 'nba':
            h *= self.dim_M**0.5
        gamma = F.leaky_relu(torch.matmul((m_dv), self.W_gamma) + self.b_gamma) #
        beta = F.leaky_relu(torch.matmul((m_dv), self.W_beta) + self.b_beta) #
        

        #neighbor mean
        i = torch.spmm(adj, h)
        i = i / degree        
        i[torch.where(degree==0)[0]] = 0.    
        assert not torch.isnan(i).any()

        # debias low-degree
        b_add = (gamma + 1) * self.W_add(i) + beta
        #b_add = self.W_add(i)

        # debias high-degree
        b_rev = (gamma + 1) * self.W_rev(i) + beta
        #b_rev = self.W_rev(i)

        mean_degree = torch.mean(degree.float())
        K = mean_degree * self.k
        R = torch.where(degree < K, torch.cuda.FloatTensor([1.]), torch.cuda.FloatTensor([0.]))


        #b_rev = b_add
        # compute constraints
        L_b = torch.sum(torch.norm((R*b_add)[idx], dim=1)) + torch.sum(torch.norm(((1-R)*b_rev)[idx], dim=1))
        L_b /= idx.shape[0]

        L_film = torch.sum(torch.norm(gamma[idx], dim=1)) + torch.sum(torch.norm(beta[idx], dim=1))
        L_film /= idx.shape[0]

        bias = self.omega * (R * b_add - (1-R) * b_rev)
        #bias = self.omega * b_add
        #bias = 0

        if self.base == 1:
            output = torch.mm(adj, h) + h + bias
            output /= (degree + 1)

        elif self.base == 2:
            
            dv = 'cuda' if x.is_cuda else 'cpu'
            N = x.size()[0]
            
            # h: N x out
            #assert not torch.isnan(h).any()

            # Self-attention on the nodes - Shared attention mechanism
            edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
            # edge: 2*D x E

            edge_e = torch.exp(-F.leaky_relu(self.a.mm(edge_h).squeeze())) #, negative_slope=0.2))
            assert not torch.isnan(edge_e).any()
            # edge_e: E

            e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
            # e_rowsum: N x 1

            edge_e = self.dropout(edge_e)
            # edge_e: E
            
            h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
            assert not torch.isnan(h_prime).any()
            
            # h_prime: N x out    
            h_prime = h_prime + bias     
            output = h_prime.div((e_rowsum+1e-5)+1)

            # h_prime: N x out
            assert not torch.isnan(output).any()
          
        elif self.base == 3:
            neighbor = torch.spmm(adj, x)
            ft_neighbor = self.weight(neighbor)
            ft_neighbor += bias 
            ft_neighbor /= (degree + 1)

            output = torch.cat([h, ft_neighbor], dim=1)

        return F.leaky_relu(output), L_b, L_film

