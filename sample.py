from __future__ import absolute_import
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data.sampler import Sampler

class RandomIdentitySampler(Sampler):
    """
    Randomly sample P identities, then for each identity,
    randomly sample K instances, therefore batch size is P*K.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py.
    Args:
        data_source (Dataset): dataset to sample from.
        num_instances (int): number of instances per identity.
    """
    def __init__(self, data_source, num_instances=4):
        # self.data_source: data set (Market1501)
        # self.num_instance: K = 4
        self.data_source = data_source
        self.num_instances = num_instances
        # self.index_dic: A dictionary, Key: identities, value: identity's images
        self.index_dic = defaultdict(list)

        # Add values (images) to keys (identities)
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)

        # self.pids: A list: [0, 1, 2, ..., 750]
        self.pids = list(self.index_dic.keys())
        # self.num_identities: 751
        self.num_identities = len(self.pids)
        # from IPython import embed
        # embed()

    def __iter__(self):
        # indices: identities after shuffle
        indices = torch.randperm(self.num_identities)
        ret = []

        # Pick 4 images for each identity
        for i in indices:
            pid = self.pids[i]
            # t: images for each identity
            t = self.index_dic[pid]
            # replace: whether images can be repeated or not
            # The images can be repeated only if the number of images of an identity is greater than 4

            if len(t) >= self.num_instances:
                replace = False
            else:
                replace = True
            t = np.random.choice(t, size=self.num_instances, replace=replace)
            ret.extend(t)
        # from IPython import embed
        # embed()

        # ret: A list for 3004 images (each identity has 4 images)
        # Each of the 4 adjacent images represents an identity
        # but the adjacent identities are not necessarily in order
        return iter(ret)

    def __len__(self):
        # P * K
        return self.num_identities * self.num_instances

# if __name__ == '__main__':
#     from data_manager import Market1501
#     dataset = Market1501(root='data')
#     sampler = RandomIdentitySampler(dataset.train, num_instances=4)