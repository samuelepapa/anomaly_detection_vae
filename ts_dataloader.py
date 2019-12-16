from ts_syntheticData import generate_timeseries, insert_anomalies, SyntheticDataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import math
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader


class RealisticDataset(Dataset):
    """Setup data for the time series data"""

    def __init__(self, data, dimensions, window_size=100, device='cpu', transform=None):
        # when window_size = 1 it means that it takes one time-step at a time
        self.device = device
        self.data = torch.tensor(data, dtype=torch.float).view(-1, dimensions)
        # the length of the time series we look at for each weight update
        self.window_size = window_size
        self.transform = transform

    def __len__(self):
        # this is the number of time serieses that are created when using a set
        # window size. If window size = len of the time series, then I have 1 time series
        # available for training
        return self.data.shape[0] - self.window_size + 1

    def __getitem__(self, idx):
        sample = self.data[idx: idx + self.window_size]
        if self.transform:
            sample = self.transform(sample)
        return sample.to(self.device), sample.to(self.device)

    def get_data(self):
        return self.data, self.data

    def has_labels(self):
        return False


def get_train_valid_test_signals(T, W, dataset_id, t_v_t_split, device, p=0.001):
    ADDANOMALIES = False
    train_valid_time = math.floor(t_v_t_split[0] * T)
    valid_test_time = math.floor((t_v_t_split[0] + t_v_t_split[1]) * T)
    if dataset_id == 0:
        signals = [
            ("sinusoid", {"frequency": 0.0019}),
            ("sinusoid", {"frequency": 0.005}),
            ("sinusoid", {"frequency": 0.003}),
            ("sinusoid", {"frequency": 0.0023}),
            ("sinusoid", {"frequency": 0.01}),
            ("sinusoid", {"frequency": 0.013}),
            ("sinusoid", {"frequency": 0.087}),
            ("sinusoid", {"frequency": 0.0007}),
            ("sinusoid", {"frequency": 0.03})
        ]
        features = len(signals)

        # generate the time series using signals
        timeseries_signals = generate_timeseries(signals, T=T)
        timeseries_signals, timeseries_labels = insert_anomalies(timeseries_signals, magnitude=0.1)

        # normalize the signal
        normalized_signals = (timeseries_signals - np.mean(timeseries_signals, axis=0)) / np.std(timeseries_signals,
                                                                                                 axis=0)
        # create train/valid/test split
        train_timeseries_signals = normalized_signals[0:train_valid_time]
        valid_timeseries_signals = normalized_signals[train_valid_time:valid_test_time]
        test_timeseries_signals = normalized_signals[valid_test_time:]

        train_timeseries_labels = timeseries_labels[0:train_valid_time]
        valid_timeseries_labels = timeseries_labels[train_valid_time:valid_test_time]
        test_timeseries_labels = timeseries_labels[valid_test_time:]

        train_dataset = SyntheticDataset([train_timeseries_signals, train_timeseries_labels], features, window_size=W,
                                         device=device)
        valid_dataset = SyntheticDataset([valid_timeseries_signals, valid_timeseries_labels], features, device=device)
        test_dataset = SyntheticDataset([test_timeseries_signals, test_timeseries_labels], features, device=device)

        print("Dataset created.")
    elif dataset_id == 1:
        # sine sequences
        signals = [
            #("sinusoid", {"frequency": 0.025}),
#            ("gp", {"kernel": 'Periodic', "p": 50}),
            ("ar", {"ar_param": [0.8, 0.15], "sigma": 1})
        ]

        # create the timeseries
        timeseries_signals = generate_timeseries(signals, T=T, noise_std=0.01,
                                                 transforms=[lambda x: x ** 2,
                                                             lambda x: x ** 2,
                                                             lambda x: 10*np.sign(x)],
                                                 transforms_std=[0.007, 0.003, 0.004])
#        timeseries_signals = generate_timeseries(signals, T=T, noise_std=0.001)
        features = timeseries_signals.shape[1]

        import matplotlib.pyplot as plt
        plt.plot(range(len(timeseries_signals)), timeseries_signals)
        timeseries_signals = (timeseries_signals - np.mean(timeseries_signals, axis=0)) / np.std(timeseries_signals,
                                                                                                 axis=0)

        timeseries_signals, timeseries_labels = insert_anomalies(timeseries_signals, magnitude=5, p=p)

        normalized_signals = (timeseries_signals - np.mean(timeseries_signals, axis=0)) / np.std(timeseries_signals,
                                                                                                 axis=0)

        train_timeseries_signals = normalized_signals[0:train_valid_time]
        valid_timeseries_signals = normalized_signals[train_valid_time:valid_test_time]
        test_timeseries_signals = normalized_signals[valid_test_time:]

        train_timeseries_labels = timeseries_labels[0:train_valid_time]
        valid_timeseries_labels = timeseries_labels[train_valid_time:valid_test_time]
        test_timeseries_labels = timeseries_labels[valid_test_time:]

        train_dataset = SyntheticDataset([train_timeseries_signals, train_timeseries_labels], features, window_size=W,
                                         device=device)
        valid_dataset = SyntheticDataset([valid_timeseries_signals, valid_timeseries_labels], features, device=device)
        test_dataset = SyntheticDataset([test_timeseries_signals, test_timeseries_labels], features, device=device)

        print("Dataset created.")
    elif dataset_id == 2:

        df_LA = pd.read_csv("LA.csv")
        features = 3
        signals = df_LA.to_numpy()
        normalized_signals = (signals - np.mean(signals, axis=0)) / np.std(signals, axis=0)
        train_dataset = RealisticDataset(normalized_signals[0:train_valid_time, :3], features, window_size=W,
                                         device=device)
        valid_dataset = RealisticDataset(normalized_signals[train_valid_time:valid_test_time, :3], features,
                                         window_size=W,
                                         device=device)
        test_dataset = RealisticDataset(normalized_signals[valid_test_time:, :3], features, window_size=W,
                                        device=device)

    # set up the DataLoader
    # This dataloader will load data with shape [batch_size, time_length, features]
    return features, \
           train_dataset, \
           valid_dataset, \
           test_dataset
