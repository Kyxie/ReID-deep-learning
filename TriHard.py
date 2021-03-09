# This code is for metric learning, Triplet Loss
# Engineer: Kunyang Xie
# Last Update: 4/3/2021


from __future__ import absolute_import
import torch
from torch import nn

class TriHardLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    Args:
        margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3):
        super(TriHardLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)

        # # Normalization
        # x = 1. * x / (torch.norm(x, 2, dim=-1, keepdim=True).expand_as(x) + 1e-12)

        # distance matrix
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()

        # mask, red & green
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_max, dist_min = [], []
        for i in range(n):
            dist_max.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_min.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_max = torch.cat(dist_max)
        dist_min = torch.cat(dist_min)
        y = torch.ones_like(dist_min)
        loss = self.ranking_loss(dist_min, dist_max, y)
        return loss

# if __name__ == '__main__':
#     target = [1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,4,4, 5,5,5,5, 6,6,6,6, 7,7,7,7, 8,8,8,8]
#     target = torch.Tensor(target)
#     feature = torch.Tensor(32, 2048)
#     a = TriHardLoss()
#     print(a.forward(feature, target))