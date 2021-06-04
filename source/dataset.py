import os
from functools import partial

import numpy as np
import pandas as pd
import torch
from PIL import Image

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms.functional as TF
import torchvision.transforms as tf

import torchvision.datasets as dset

class MNISTRot(Dataset):

    def __init__(self, filepath, range_data=None, angles=[-45, -25, 0, 25, 45], random=True, reshape=True, angle_fixed=None):
        X, y = torch.load(filepath)

        if range_data is not None:
            if random:
                idx = np.random.choice(np.arange(X.shape[0]), range_data, replace=True)
            else:
                idx = np.arange(range_data)

            X = X[idx, ...]
            y = y[idx]

        self.X = X
        self.y = y

        self.angles = np.array(angles)
        self.reshape = reshape
        self.angle_fixed = angle_fixed

    @classmethod
    def from_dict(cls, config_data, type='train'):
        """
        Create a dataset from a config dictionary
        :param: config_dict : configuration dictionary
        """

        filepath = config_data[type]

        range_data = None
        if ('range_data' in config_data):
            range_data = config_data['range_data']

        angle_fixed = None
        if 'angle_fixed' in config_data:
            angle_fixed = config_data['angle_fixed']

        reshape = config_data['reshape']

        return cls(filepath, range_data=range_data, reshape=reshape, angle_fixed=angle_fixed)

    def __getitem__(self, idx):
        """
        Randomly rotate the digit with an angle +- 25, +- 45 or 0
        :param idx:
        :return:
        """
        if self.angle_fixed is None:
            angle_dx = np.random.choice(np.arange(self.angles.shape[0]), 1)
        else:
            angle_dx= self.angle_fixed

        angle = self.angles[angle_dx]

        x = self.X[idx, ...]
        x = tf.ToPILImage()(x)
        x = TF.rotate(x, angle)
        x = tf.ToTensor()(x)

        if self.reshape:
            x = x.reshape(28 * 28)

        sensitive = torch.zeros(self.angles.shape[0])
        sensitive[angle_dx] = 1


        return {'input': x, 'target': x, 'outcome': self.y[idx].long(), 'sensitive': sensitive}

    def __len__(self):
        return self.X.shape[0]

class BinarizedMNISTRot(MNISTRot):

    def __getitem__(self, idx):
        """
        Randomly rotate the digit with an angle +- 25, +- 45 or 0. The data is resized to 32 by 32 and
        binarized using bernouilli draws.
        :param idx:
        :return:
        """
        if self.angle_fixed is None:
            angle_dx = np.random.choice(np.arange(self.angles.shape[0]), 1)
        else:
            angle_dx= self.angle_fixed

        angle = self.angles[angle_dx]

        x = self.X[idx, ...]
        x = tf.ToPILImage()(x)
        x = TF.rotate(x, angle)
        x = tf.Pad(2)(x)
        x = tf.ToTensor()(x)

        x = torch.bernoulli(x)

        sensitive = torch.zeros(self.angles.shape[0])
        sensitive[angle_dx] = 1


        return {'input': x, 'target': x, 'outcome': self.y[idx].long(), 'sensitive': sensitive}


class CropCelebA64(object):
    """ This class applies cropping for CelebA64. This is a simplified implementation of:
    https://github.com/andersbll/autoencoding_beyond_pixels/blob/master/dataset/celeba.py
    """
    def __call__(self, pic):
        new_pic = pic.crop((15, 40, 178 - 15, 218 - 30))
        return new_pic

    def __repr__(self):
        return self.__class__.__name__ + '()'

class CelebA64(Dataset):
    """
    Dataset class with len and get methods
    The dataset is initialized by a path to an index file and
    is represented by a index table
    """

    def __init__(self, indexpath, range_data=None, resize=True):
        self.indextable = pd.read_csv(os.path.join(indexpath, 'index.csv'))
        self.indextable.set_index(np.arange(len(self.indextable)), inplace=True)
        self.indextable['file'] = self.indextable.file.apply(lambda file: os.path.join(indexpath, file))
        if resize:
            self.transform = tf.Compose([tf.Resize(64), tf.ToTensor()])
        else:
            self.transform = tf.Compose([tf.ToTensor()])

        if range_data is not None:
            range_data = min(len(self.indextable), range_data)
            index = np.random.choice(self.indextable.index, range_data, replace=False)
            self.indextable = self.indextable.loc[index, :]
            self.indextable.set_index(np.arange(len(self.indextable)), inplace=True)

    @classmethod
    def from_dict(cls, config_data, type='train'):
        """
        Create a dataset from a config dictionary
        """

        filepath = config_data[type]

        range_data = None
        if ('range_data' in config_data):
            range_data = config_data['range_data']

        return cls(filepath, range_data=range_data)

    def __getitem__(self, idx):
        """
        Overload __getitem__ with idx being an index value in self.indextable
        :param idx:
        :return: a torch tensor (C, W, H)
        """

        img = Image.open(self.indextable.loc[idx, 'file'])
        img_data = self.transform(img)

        sensitive = (self.indextable.loc[idx, 'Male'] + 1) / 2
        s = torch.zeros(2)
        if sensitive == 1:
            s[0] = 1
        else:
            s[1] = 1

        return {'input': img_data, 'target': img_data, 'sensitive': s.float()}

    def __len__(self):
        """
        Overload length method for dataset
        :return: len(self.indextable)
        """
        return len(self.indextable)


class CelebA(Dataset):
    """
    Dataset class with len and get methods
    The dataset is initialized by a path to an index file and
    is represented by a index table
    """

    def __init__(self, root, range_data=None, size=256, split='train'):

        self.base_folder = "celeba"
        self.root = root

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }

        split_ = split_map[split]

        fn = partial(os.path.join, self.root, self.base_folder)
        splits = pd.read_csv(fn("list_eval_partition.csv"), index_col='image_id')
        identity = pd.read_csv(fn("identity_celeba.csv"), index_col='image_id')
        attr = pd.read_csv(fn("list_attr_celeba.csv"), index_col='image_id')

        mask = slice(None) if split_ is None else (splits['partition'] == split_)

        self.attr = attr
        self.filename = splits[mask].index.values
        self.identity = identity

        self.transform = tf.Compose([tf.Resize(size), tf.CenterCrop(size), tf.ToTensor(),
                                #tf.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                ]
                               )

        if range_data is not None:
            range_data = min(len(self.filename), range_data)
            index = np.random.choice(len(self.filename), range_data, replace=False)
            self.filename = self.filename[index]

    @classmethod
    def from_dict(cls, config_data, type='train'):
        """
        Create a dataset from a config dictionary
        """

        filepath = config_data[type]

        range_data = None
        if ('range_data' in config_data):
            range_data = config_data['range_data']

        return cls(filepath, range_data=range_data, split=type)

    def __getitem__(self, idx):
        """
        Overload __getitem__ with idx being an index value in self.indextable
        :param idx:
        :return: a torch tensor (C, W, H)
        """
        image_id = self.filename[idx]
        img = Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", image_id))
        img_data = self.transform(img)

        sensitive = self.attr.loc[image_id, 'Male']
        s = torch.zeros(2)
        if sensitive == 1:
            s[0] = 1
        else:
            s[1] = 1


        return {'input': img_data, 'target': img_data, 'sensitive': s.float()}

    def __len__(self):
        """
        Overload length method for dataset
        :return: len(self.indextable)
        """
        return len(self.filename)


class CIFAR10(object):

    def __init__(self, root, type='train'):
        cifar_transform = tf.Compose([tf.RandomHorizontalFlip(), tf.ToTensor()])

        if type == 'train':
            self.cifar10 = dset.CIFAR10(root=root, train=True, transform=cifar_transform)
        else:
            self.cifar10 = dset.CIFAR10(root=root, train=False, transform=cifar_transform)

    @classmethod
    def from_dict(cls, config_data, type='train'):
        """
        Create a dataset from a config dictionary
        """

        root = config_data['root']
        return cls(root, type=type)

    def __getitem__(self, item):

        img, y = self.cifar10[item]
        s = torch.zeros(2)

        return {'input': img, 'target': img, 'outcome': s, 'sensitive': s}

    def __len__(self):
        return len(self.cifar10)


class RepDataset(Dataset):
    """
    Generate represenation using the generator function that takes
    data from dset and compute generator(dset)
    """

    def __init__(self, dset, generator, device='cpu', batch_size=128, threshold=1, shuffle=True):
        super().__init__()
        self.dset = dset
        self.generator = generator
        self.device = device
        self.col_id = None

        input_list = []
        outcome_list = []
        sensitive_list = []
        z_list = []

        data_loader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle)

        for batch in data_loader:
            z = self.generate(batch['input'].to(device), threshold=threshold, s=batch['sensitive'].to(device))
            input_list.append(batch['input'].cpu())
            z_list.append(z.cpu())

            outcome_list.append(batch['outcome'].cpu())
            sensitive_list.append(batch['sensitive'].cpu())

        self.outcome = torch.cat(outcome_list, 0)
        self.sensitive = torch.cat(sensitive_list, 0)
        self.data = torch.cat(z_list, 0)
        self.input = torch.cat(input_list)

        self.zdim = z.shape[-1]

    def generate(self, x, threshold=0.5, s=None):
        """
        return represenation from x
        :return:
        """
        beta = torch.tensor([threshold]).expand_as(s[:, 0]).float()
        beta = beta.to(self.device)
        _, q, m, _, _ = self.generator.net.forward(x, s, beta)
        z = q.reshape(x.shape[0], -1).cpu().detach()
        return z

    def __getitem__(self, idx):
        """
        Overload __iter__ with idx being an index value
        it generates as data the output of the generator encoder

        :return: an iterable
        """
        x = self.input[idx, ...]
        y = self.outcome[idx, ...]
        s = self.sensitive[idx, ...]
        z = self.data[idx, ...]

        return {'input_mean': z, 'target': y, 'sensitive': s, 'input': x}

    def __len__(self):
        """
        :return: the length of data
        """
        return len(self.dset)


class CelebA64(Dataset):
    """
    Dataset class with len and get methods
    The dataset is initialized by a path to an index file and
    is represented by a index table
    """

    def __init__(self, indexpath, range_data=None, transform=None):
        self.indextable = pd.read_csv(os.path.join(indexpath, 'index.csv'))
        self.indextable.set_index(np.arange(len(self.indextable)), inplace=True)
        self.indextable['file'] = self.indextable.file.apply(lambda file: os.path.join(indexpath, file))
        self.transform = tf.Compose([tf.Resize(64), tf.ToTensor()])

        if range_data is not None:
            range_data = min(len(self.indextable), range_data)
            index = np.random.choice(self.indextable.index, range_data, replace=False)
            self.indextable = self.indextable.loc[index, :]
            self.indextable.set_index(np.arange(len(self.indextable)), inplace=True)

    @classmethod
    def from_dict(cls, config_data, type='train'):
        """
        Create a dataset from a config dictionary
        """

        filepath = config_data[type]

        range_data = None
        if ('range_data' in config_data):
            range_data = config_data['range_data']

        return cls(filepath, range_data=range_data)

    def __getitem__(self, idx):
        """
        Overload __getitem__ with idx being an index value in self.indextable
        :param idx:
        :return: a torch tensor (C, W, H)
        """

        img = Image.open(self.indextable.loc[idx, 'file'])
        img_data = self.transform(img)

        sensitive = (self.indextable.loc[idx, 'Male'] + 1) / 2
        s = torch.zeros(2)
        if sensitive == 1:
            s[0] = 1
        else:
            s[1] = 1

        return {'input': img_data, 'target': img_data, 'sensitive': s.float()}

    def __len__(self):
        """
        Overload length method for dataset
        :return: len(self.indextable)
        """
        return len(self.indextable)