import os, sys
import pickle as pkl
import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.stats import pearsonr

import pandas as pd
from collections import Counter, OrderedDict
import matplotlib.pyplot as plt
import torch
import json
from scipy.spatial import distance_matrix

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    rowsum = np.where(rowsum==0, 1, rowsum)
    r_inv = np.power(rowsum, -1).flatten()
    #r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)    
    mx = r_mat_inv.dot(mx)
    return mx

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    rowsum = np.where(rowsum==0, 1, rowsum)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    
    if sp.issparse(features):
        return features.todense()
    else:
        return features


def convert_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def feature_norm(features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    return 2*(features - min_values).div(max_values-min_values) - 1



class Dataset(object):
    def __init__(self):
        self.adj = None
        self.feat = None
        self.idx = None
        self.labels = None
        self.degree = None
        self.deg_labels = None
        self.gt = None
        self.edge = None

    def get_info(self):
        return

    def to_tensor(self):
        self.feat = torch.FloatTensor(self.feat)
        self.labels = torch.LongTensor(self.labels)
        self.degree = torch.LongTensor(self.degree)
        self.deg_labels = torch.FloatTensor(self.deg_labels)
        if self.gt is not None:
            self.gt = torch.LongTensor(self.gt)
        return

    def to_cuda(self):
        #self.edge_index = self.edge_index.cuda()
        self.adj = self.adj.cuda()
        self.feat = self.feat.cuda()
        self.labels = self.labels.cuda()
        self.degree = self.degree.cuda()
        self.deg_labels = self.deg_labels.cuda()
        if self.gt is not None:
            self.gt = self.gt.cuda()
        if self.edge is not None:
            self.edge = self.edge.cuda()
        return
    

    def get_str_info(self, degree):
        str_info = np.linalg.matrix_power(self.adj, degree)
        return np.sum(str_info, axis=1) 


    def get_statistics(self):
        
        self.to_tensor()
       
        degree = np.squeeze(self.degree)

        labels = []
        mean = []
        for i in np.unique(self.labels):
            tmp = np.where(self.labels == i)[0]            
            d = torch.mean(degree[tmp], dtype=float).item()
            labels.append(i)
            mean.append(d)

        labels = np.asarray(labels)
        mean = np.asarray(mean)

        std = np.std(mean)
        avg_deg = np.mean(degree.numpy()) #np.mean(self.degree)

        print('Std: {:.4f}, AvgDe: {:.4f}, Std/AvgDe: {:.4f}'.format(std, avg_deg, std/avg_deg))



class Squirrel(Dataset):
    def __init__(self, path, norm=False, use_dw=False, degree=2):
        super(Squirrel, self).__init__()
        self.path = path + 'Squirrel/'

        adj, feat, labels = self.load_data()

        self.adj = adj.todense()
        self.degree = np.sum(self.adj, axis=1)
        self.max_degree = int(self.degree.max())

        self.group1 = self.get_str_info(degree=1)
        self.group2 = self.get_str_info(degree=2)

        if norm:
            self.adj = normalize(self.adj + np.eye(self.adj.shape[0]))

        self.feat = feat
        self.labels = labels
        self.deg_labels = self.degree / self.max_degree

        idx = np.arange(feat.shape[0])
        np.random.seed(0)
        np.random.shuffle(idx)
        p = int(0.6 * idx.shape[0])
        pt = int(0.8 * idx.shape[0])
        self.idx_train = idx[:p]
        self.idx_val = idx[p:pt]
        self.idx_test = idx[pt:]

        
    def load_data(self):
        with open("{}node_feature_label.txt".format(self.path), 'rb') as f:
            clean_lines = (line.replace(b'\t',b',') for line in f)
            load_features = np.genfromtxt(clean_lines, skip_header=1, dtype=np.float32, delimiter=',')
        
        idx = load_features[load_features[:,0].argsort()] # sort regards to ascending index
        labels = idx[:,-1]
        features = idx[:,1:-1]

        edges = np.genfromtxt("{}graph_edges.txt".format(self.path), dtype=np.int32)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(idx.shape[0], idx.shape[0]),
                            dtype=np.float32)
        
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = adj.tolil()
        
        for i in range(adj.shape[0]):
            adj[i, i] = 0.

        features = preprocess_features(features)
        return adj, features, labels


class Chameleon(Dataset):
    def __init__(self, path, norm=False, use_dw=False, degree=2):
        super(Chameleon, self).__init__()
        self.path = path + 'Chameleon/'

        adj, feat, labels = self.load_data()

        self.adj = adj.todense()
        self.degree = np.sum(self.adj, axis=1)
        self.max_degree = int(self.degree.max())

        self.group1 = self.get_str_info(degree=1)
        self.group2 = self.get_str_info(degree=2)

        if norm:
            self.adj = normalize(self.adj + np.eye(self.adj.shape[0]))

        self.feat = feat
        self.labels = labels
        self.deg_labels = self.degree / self.max_degree

        idx = np.arange(feat.shape[0])
        np.random.seed(0)
        np.random.shuffle(idx)
        p = int(0.6 * idx.shape[0])
        pt = int(0.8 * idx.shape[0])
        self.idx_train = idx[:p]
        self.idx_val = idx[p:pt]
        self.idx_test = idx[pt:]

       

    def load_data(self):
        with open("{}node_feature_label.txt".format(self.path), 'rb') as f:
            clean_lines = (line.replace(b'\t',b',') for line in f)
            load_features = np.genfromtxt(clean_lines, skip_header=1, dtype=np.float32, delimiter=',')
        
        idx = load_features[load_features[:,0].argsort()] # sort regards to ascending index
        labels = idx[:,-1]
        features = idx[:,1:-1]

        edges = np.genfromtxt("{}graph_edges.txt".format(self.path), dtype=np.int32)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(idx.shape[0], idx.shape[0]),
                            dtype=np.float32)
        
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = adj.tolil()
        
        for i in range(adj.shape[0]):
            adj[i, i] = 0.

        features = preprocess_features(features)
        return adj, features, labels



class EMNLP(Dataset):
    def __init__(self, path, norm=False, use_dw=False, degree=2):
        super(EMNLP, self).__init__()

        self.path = path + 'EMNLP/'

        adj, feat, labels = self.load_data()

        self.adj = adj.todense()
        self.degree = np.sum(self.adj, axis=1)
        self.max_degree = int(self.degree.max())

        self.group1 = self.get_str_info(degree=1)
        self.group2 = np.load(self.path + 'd2.npy')
        #np.save(self.path + 'd2.npy', self.group2)

        if norm:
            self.adj = normalize(self.adj + np.eye(self.adj.shape[0]))

        self.feat = feat
        self.labels = labels
        self.deg_labels = self.degree / self.max_degree

        idx = np.arange(feat.shape[0])
        np.random.seed(0)
        np.random.shuffle(idx)
        p = int(0.6 * idx.shape[0])
        pt = int(0.8 * idx.shape[0])
        self.idx_train = idx[:p]
        self.idx_val = idx[p:pt]
        self.idx_test = idx[pt:]

       
    def load_data(self):
        adj = sp.load_npz(self.path + 'emnlp-adj.npz')
        feat = np.load(self.path + 'emnlp-x.npy')
        target = np.load(self.path + 'emnlp-y.npy')

        p = np.mean(target)
        label = np.where(target < p, 0, 1)
        
        count = Counter(label)
        # print(count)
        # exit()

        return adj, feat, label



DATASETS = {
    'squirrel': Squirrel,
    'chameleon': Chameleon,
    'emnlp': EMNLP
}


def get_dataset(name, path='data/', norm=None, use_dw=None, degree=2):
    if name not in DATASETS:
        raise ValueError("Dataset is not supported")
    return DATASETS[name](path=path, norm=norm, use_dw=use_dw, degree=degree)


if __name__ == "__main__":
    dataset = get_dataset('chameleon')
