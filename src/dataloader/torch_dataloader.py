from __future__ import print_function, division
import torch
from torch.utils.data import Dataset
import numpy as np

from src.datareader.read_segment import ReadSegment
from src.features.feature_functions import Features
# from utils.file_io import get_signal_files


class CustomDataset(Dataset):

    def __init__(self, users, stride=25, transform=None, window=50, extract_feat=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
                :param extract_feat:
        """
        reader = ReadSegment(users)
        self.segments, self.labels = reader.segment(window, stride)
        feature_extractor = Features()

        if extract_feat:
            self.features = feature_extractor.get_features(self.segments)
        else:
            self.features = np.transpose(self.segments, [1, 0, 2])
        print("Dataset created with shape -> Features: ", self.features.shape, "Labels: ", self.labels.shape)

        self.features = torch.from_numpy(self.features).type(torch.float)
        self.labels = torch.from_numpy(self.labels).type(torch.long)
        self.transform = transform

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.features[idx]
        y = self.labels[idx]
        sample = {'x': x, 'y': y}
        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == "__main__":
    hapt_dataset = CustomDataset([1, 2, 3])
    # , 18, 6, 4, 25, 23
    sample = hapt_dataset[1]
    print(sample['x'].shape, sample['y'].shape)
    print(len(hapt_dataset))
