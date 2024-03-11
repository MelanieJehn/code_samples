import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class ClassifierDataset(Dataset):

    def __init__(self, data, transform=None):
        """
        Args:
            data (string or dataframe): Path to the csv file or dataframe containing token data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        # Read the file and store the content in a pandas DataFrame
        if isinstance(data, str):
            self.df = pd.read_csv(data)
        else:
            self.df = data

        # Padding
        self.max_len = 0
        unpadded = []
        for index, row in self.df.iterrows():
            tokens = row['tokens']
            a_list = tokens[1:-1].split(', ')
            map_object = map(int, a_list)
            tokens = list(map_object)

            unpadded.append(tokens)

            if len(tokens) > self.max_len:
                self.max_len = len(tokens)

        padded = np.array([i + [0] * (self.max_len - len(i)) for i in unpadded])
        self.padded = np.array(padded)

        # Attention mask
        self.attention_mask = np.where(self.padded != 0, 1, 0)

        labels = row['label']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample = (self.padded[idx], self.attention_mask[idx], row['label'])

        if self.transform:
            sample = self.transform(sample)
        return sample


class ContextDataset(Dataset):

    def __init__(self, data, transform=None):
        """
        Args:
            data (string or dataframe): Path to the csv file or dataframe containing token data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        # Read the file and store the content in a pandas DataFrame
        if isinstance(data, str):
            self.df = pd.read_csv(data)
        else:
            self.df = data

        # Padding
        self.max_len = 0
        unpadded = []
        for index, row in self.df.iterrows():
            tokens = row['tokens']
            a_list = tokens[1:-1].split(', ')
            map_object = map(int, a_list)
            tokens = list(map_object)

            unpadded.append(tokens)

            if len(tokens) > self.max_len:
                self.max_len = len(tokens)

        padded = np.array([i + [0] * (self.max_len - len(i)) for i in unpadded])
        self.padded = np.array(padded)

        # Attention mask
        self.attention_mask = np.where(self.padded != 0, 1, 0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = [float(x) for x in row['out'][1:-1].split(', ')]
        sample = (self.padded[idx], self.attention_mask[idx], label)

        if self.transform:
            sample = self.transform(sample)
        return sample