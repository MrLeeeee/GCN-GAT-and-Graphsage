# conding: utf-8
import numpy as np
from torch.utils import data
from config import opt
from tqdm import tqdm
import torch
import random
import scipy.sparse as sp


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))             # 对每一个特征进行归一化
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_data(opt):
    print("Loading {} dataset..." .format(opt.network))
    idx_features_labels = np.genfromtxt("./data/{}.content" .format(opt.network), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)  #特征
    labels = encode_onehot(idx_features_labels[:, -1]) # 类别的one-hot编码

    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("./data/{}.cites".format(opt.network), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape) # 编码到编号的转换
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32) #构建邻接矩阵

    # 构建成对称矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    labels_map = {i: [] for i in range(labels.shape[1])}
    labels = np.where(labels)[1]
    for i in range(labels.shape[0]):
        labels_map[labels[i]].append(i)
    for ele in labels_map:
        random.shuffle(labels_map[ele])
    idx_train = list()
    idx_val = list()
    idx_test = list()
    for ele in labels_map:
        idx_train.extend(labels_map[ele][0:int(opt.train_rate * labels.shape[0])])
        idx_val.extend(labels_map[ele][int(opt.train_rate * labels[0]):int((opt.train_rate + opt.val_rate) * labels.shape[0])])
        idx_test.extend(labels_map[ele][int((opt.train_rate + opt.val_rate) * labels.shape[0]):])
    features = torch.FloatTensor(np.array(features.todense()))       # 类型为torchTensor是为了在神经网络中的乘法
    adj = torch.FloatTensor(np.array(adj.todense()))

    return adj, features, labels, idx_train, idx_val, idx_test

class Dataload(data.Dataset):

    def __init__(self, labels, id):
        self.data = id
        self.labels = labels

    def __getitem__(self, index):
        return index, self.labels[index]

    def __len__(self):
        return self.data.__len__()