from __future__ import print_function, division
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from segmentation.get_segments import get_segments
from features.feature_functions import get_features
from utils.file_io import get_signal_files

class HAPTDataset(Dataset):

    def __init__(self, users, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.users = users
        files = get_signal_files(self.users)
        self.segments, self.labels = SegmentFiles(files)
        self.features = get_features(self.segments)
        self.transform = transform

    def __len__(self):
        return (self.features.shape[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.features[idx, :]
        y = self.labels[idx, :]
        sample = {'x': x, 'y': y}

        if self.transform:
            sample = self.transform(sample)
        return sample