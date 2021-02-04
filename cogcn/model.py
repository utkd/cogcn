import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import GraphConvolution

class GCNAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNAE, self).__init__()
        self.encgc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.encgc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)

        self.decgc1 = GraphConvolution(hidden_dim2, hidden_dim1, dropout, act=F.relu)
        self.decgc2 = GraphConvolution(hidden_dim1, input_feat_dim, dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.encgc1(x, adj)
        hidden2 = self.encgc2(hidden1, adj)
        return hidden2

    def decode(self, hidden, adj):
        hidden1 = self.decgc1(hidden, adj)
        recon = self.decgc2(hidden1, adj)
        return recon

    def forward(self, x, adj):
        enc = self.encode(x, adj)
        dec = self.decode(enc, adj)
        return dec, enc

class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj
