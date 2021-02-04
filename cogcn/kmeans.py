import sys
import torch
import torch.nn as nn

from sklearn.cluster import KMeans

class Clustering(object):

    def __init__(self, K, n_init=5, max_iter=250):
        self.K = K
        self.n_init = n_init
        self.max_iter = max_iter

        self.u = None
        self.M = None

    def cluster(self, embed):
        embed_np = embed.detach().cpu().numpy()
        clustering = KMeans(n_clusters=self.K, n_init=self.n_init, max_iter=self.max_iter)
        clustering.fit(embed_np)

        self.M = clustering.labels_
        self.u = self._compute_centers(self.M, embed_np)

    def get_loss(self, embed):
        loss = torch.Tensor([0.])
        #TODO: This may be slightly inefficient, we can fix it later to use matrix multiplications
        for i, clusteridx in enumerate(self.M):
            x = embed[i]
            c = self.u[clusteridx]
            difference = x - c
            err = torch.sum(torch.mul(difference, difference))
            loss += err

        return loss

    def get_membership(self):
        return self.M

    def _compute_centers(self,labels, embed):
        """
        sklearn kmeans may not give accurate cluster centers in some cases (see doc), so we compute ourselves
        """
        clusters = {}
        for i,lbl in enumerate(labels):
            if clusters.get(lbl) is None:
                clusters[lbl] = []
            clusters[lbl].append(torch.FloatTensor(embed[i]))
        
        centers = {}
        for k in clusters:
            all_embed = torch.stack(clusters[k])
            center = torch.mean(all_embed, 0)
            centers[k] = center
        
        return centers