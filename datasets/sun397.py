import os
import os.path
from glob import glob

import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms


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
        line = os.path.join(root, line[1:].strip())
        # print(line)
        assert os.path.isfile(line)
        images.append(line)
        for classname in class_to_idx:
            if f'/{classname}/' in line:
                labels.append(class_to_idx[classname])
                break

    return images, labels


class SUN397(data.Dataset):
    def __init__(self, root, split=True, transform=None, target_transform=None, download=None):
        classes, class_to_idx = find_classes(os.path.join(root, 'ClassName.txt'))
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.train = (split=='train') 
        self.transform = transform
        self.target_transform = target_transform
        self.split = '01'
        path = os.path.join(root, 'Training_01' + '.txt')
        # if split == 'train':
            
        # elif split == 'val':
        #     path = os.path.join(root, 'val' + '.txt')
        # elif split == 'test':
        #     path = os.path.join(root, 'test' + '.txt')
        # else:
        #     raise("Wrong split!")

        # path = os.path.join(root, 'img_path' + '.txt') # all data
        self.images, self.labels = make_dataset(path, root, class_to_idx)
        assert (len(self.images) == len(self.labels))

    def __getitem__(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _label = self.labels[index]
        if self.transform is not None:
            _img = self.transform(_img)
        if self.target_transform is not None:
            _label = self.target_transform(_label)

        return _img, _label

    def __len__(self):
        return len(self.images)
