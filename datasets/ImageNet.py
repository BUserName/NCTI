import os
import os.path
from glob import glob

import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms

from torchvision.datasets.folder import pil_loader, accimage_loader, default_loader

def find_classes(file_path):
    with open(file_path, 'r') as f:
        classes = [d[3:] for d in f.read().splitlines()]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(path, root, class_to_idx):
    images = []
    labels = []
    for line in open(path, 'r'):
        # print(root)
        line = os.path.join(root, line[2:].strip())
        # print(line)
        assert os.path.isfile(line)
        images.append(line)
        for classname in class_to_idx:
            if f'/{classname}/' in line:
                labels.append(class_to_idx[classname])
                break

    return images, labels


class ImageNet(data.Dataset):
    def __init__(self, root, split, transform=None, target_transform=None, download=False, loader=default_loader):
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        # create_caltech101_splits('./data')   
        # remove clutter class directory
        #clutter_dir = os.path.join(root, 'caltech101', '101_ObjectCategories', 'BACKGROUND_Google')
        #if os.path.exists(clutter_dir):
        #    shutil.rmtree(clutter_dir, ignore_errors=True)
        # find indices for split
        print(os.path.join(root, 'filename.txt'))
        with open(os.path.join(root, 'filename.txt'), 'r') as f:
            
            self.samples = [(os.path.join(root, line.strip()), int(line.split('/')[0])) for line in f.readlines()]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

