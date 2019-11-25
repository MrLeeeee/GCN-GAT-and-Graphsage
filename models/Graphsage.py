import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from models.BasicModel import BasicModule

class PyGraphsage(BasicModule):
    def __init__(self, nfeat, nhid, nclass):
        super(PyGraphsage, self).__init__()
        self.model_name = 'PyGraphsage'
        self.droput = nn.Dropout()
        self.sage1 = Graphsage(nfeat, nhid)
        self.sage2 = Graphsage(nhid, nhid)
        self.att = nn.Linear(nhid, nclass)

    def forward(self, input, adj):
        hid1 = self.sage1(input, adj)
        hid1 = self.droput(hid1)
        hid2 = self.sage2(hid1, adj)
        out = self.att(hid2)
        return F.log_softmax(out, dim=1)

class Graphsage(nn.Module):
    def __init__(self, infeat, outfeat):
        super(Graphsage, self).__init__()
        self.infeat = infeat
        self.model_name = 'Graphsage'
        self.W = nn.Parameter(torch.zeros(size=(2 * infeat, outfeat)))
        self.bias = nn.Parameter(torch.zeros(outfeat))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        h1 = torch.mm(adj, input)
        degree = adj.sum(axis=1).repeat(self.infeat, 1).T
        h1 = h1/degree
        h1 = torch.cat([input, h1], dim=1)
        h1 = torch.mm(h1, self.W)
        return h1