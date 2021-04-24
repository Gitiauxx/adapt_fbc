import numpy as np
import pandas as pd
import torch

from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as tf

class MNISTRot(Dataset):

    def __init__(self, filepath, range_data=None, angles=[-45, 0, 45], random=True, reshape=True, angle_fixed=None):
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