import sys
import torch
import torch.nn as nn
import torch.nn.modules.loss
import torch.nn.functional as F

from sklearn.cluster import KMeans

def compute_attribute_loss(lossfn, features, recon, outlier_wt):
    loss = lossfn(features, recon)
    loss = loss.sum(dim=1)

    outlier_wt = torch.log(1/outlier_wt)

    attr_loss = torch.sum(torch.mul(outlier_wt, loss))

    return attr_loss

def compute_structure_loss(adj, embed, outlier_wt):
    # to compute F(x_i).F(x_j)
    embeddot = torch.mm(embed, torch.transpose(embed, 0, 1))

    adj_tensor = adj.to_dense()

    # compute A_ij - F(x_i)*F(x_j)
    difference = adj_tensor - embeddot
    # square difference and sum
    loss = torch.sum(torch.mul(difference, difference), dim=1)

    outlier_wt = torch.log(1/outlier_wt)

    struct_loss = torch.sum(torch.mul(outlier_wt, loss))

    return struct_loss

def update_o1(adj, embed):
    # to compute F(x_i).F(x_j)
    embed = embed.data
    embeddot = torch.mm(embed, torch.transpose(embed, 0, 1))

    adj_tensor = adj.to_dense()

     # compute A_ij - F(x_i)*F(x_j)
    difference = adj_tensor - embeddot
    # square difference and sum
    error = torch.sum(torch.mul(difference, difference), dim=1)

    # compute the denominator
    normalization_factor = torch.sum(error)

    # normalize the errors
    o1 = error/normalization_factor

    return o1

def update_o2(features, recon):
    features = features.data
    recon = recon.data
    # error = x - F(G(x))
    error = features - recon
    # error now = (x - F(G(x)))^2, summed across dim 1
    error = torch.sum(torch.mul(error, error), dim=1)

    # compute the denominator
    normalization_factor = torch.sum(error)

    # normalize the errors
    o2 = error/normalization_factor

    return o2
