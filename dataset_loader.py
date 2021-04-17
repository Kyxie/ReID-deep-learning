from __future__ import print_function, absolute_import
from PIL import Image
import os.path as osp
from torch.utils.data import Dataset

# Check image
def read_image(img_path):
    got_img = False
    if not osp.exists(img_path):
        print("'{}' does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("Does not read image")
            pass
    return img

class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid

# if __name__ == '__main__':
#     import data_manager
#     dataset = data_manager.init_img_dataset(root='data', name='UESTC')
#     gallery_loader = ImageDataset(dataset.gallery)
#     from IPython import embed
#     embed()