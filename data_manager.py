from __future__ import print_function, absolute_import
import os.path as osp
import glob
import re

class Market1501(object):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """

    def __init__(self, root='data'):
        # Paths of Data Sets
        self.dataset_dir = osp.join(root, 'market1501')
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()

        # Train Set
        train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir, relabel=True)
        # train: A list which templete is: (image path, person id, camera id)
        # num_train_pids: Total person number in train set
        # num_train_imgs: Total image number in train set
        # embed()

        # Test Set
        query, num_query_pids, num_query_imgs = self._process_dir(self.query_dir, relabel=False)
        gallery, num_gallery_pids, num_gallery_imgs = self._process_dir(self.gallery_dir, relabel=False)

        # Total
        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        print("=> Market1501 loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

        # embed()

    def _check_before_run(self):
        # Check if all paths are available
        if not osp.exists(self.dataset_dir):
            print("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            print("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            print("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            print("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        # Find the file name, Person ID, Camera ID, Picture Quantity
        # dir_path: File path
        # relable: Some person id are not in train set, we need to delete them

        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))   # Add jpg file in dir_path
        # embed()

        # Delete person ID which do not in train set
        pattern = re.compile(r'([-\d]+)_c(\d)')
        pid_container = set()   # pid_container: person ID which are in train set
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:   # Rubbish Data
                continue
            pid_container.add(pid)
        # embed()

        # Rearrange
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        # embed()

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:  # Rubbish Data
                continue
            assert 0 <= pid <= 1501
            assert 1 <= camid <= 6
            camid -= -1 # camera id minus 1, start from 0
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, pid, camid))
        num_pids = len(pid_container)   # How many id in train set (751)
        num_imgs = len(img_paths) # How many images
        return dataset, num_pids, num_imgs

__img_factory = {
    'market1501': Market1501
}

def init_img_dataset(name, **kwargs):
    if name not in __img_factory.keys():
        raise KeyError(
            "Invalid dataset, got '{}', but expected to be one of {}".format(name, __img_factory.keys()))
    return __img_factory[name](**kwargs)

# if __name__ == '__main__':
#     data = Market1501()
#     data.__init__()