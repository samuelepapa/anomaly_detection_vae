from ts_syntheticData import generate_timeseries, insert_anomalies, SyntheticDataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import math
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader


# RealisticDataset
# helper class used to load the realistic weather dataset.
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

# arguments:
#   - T: the length of the sequence to load.
#   - scenario: the scenario selected. 1: simple dataset, 2: correlated dataset, 3: weather dataset.
def load_data(scenario, T):
    if scenario == 0:
        signals = [
            ("sinusoid", {"frequency": 0.47}),
            ("ar", {"ar_param": [0.8, 0.15], "sigma": 1}),
            ("sinusoid", {"frequency": 0.5}),
            ("sinusoid", {"frequency": 0.3}),
            ("sinusoid", {"frequency": 0.23}),
            ("sinusoid", {"frequency": 0.1}),
            ("sinusoid", {"frequency": 0.3}),
            ("sinusoid", {"frequency": 0.87}),
            ("sinusoid", {"frequency": 0.07}),
            ("sinusoid", {"frequency": 0.03})
        ]
        # generate the time series using signals
        signals = generate_timeseries(signals, T=T, noise_std=0.01)

    elif scenario == 1:
        # sine sequences
        signals = [
            ("sinusoid", {"frequency": 0.023, "amplitude": 2}),
            # ("sinusoid", {"frequency": 0.05, "amplitude": 2}),
            #            ("gp", {"kernel": 'Periodic', "p": 50}),
            ("ar", {"ar_param": [0.8, 0.15], "sigma": 1})
        ]

        # create the timeseries
        signals = generate_timeseries(signals, T=T, noise_std=0.01,
                                      transforms=[  # lambda x: 5 * np.sin(x),
                                          lambda x: x ** 3],
                                      #            #lambda x: 5 * np.sign(x)],
                                      transforms_std=[0.07, 0.03, 0.04])
        #        timeseries_signals = generate_timeseries(signals, T=T, noise_std=0.001)
    elif scenario == 2:
        df_LA = pd.read_csv("LA.csv")
        signals = df_LA.to_numpy()[:T, :]

    elif scenario == 3:
        df_LA = pd.read_csv("LA.csv")
        signals = df_LA.to_numpy()[:T, :]
    else:
        raise ValueError("Scenario not recognized, it should be 0, 1, 2 or 3.")
    return signals


# splits the data in different dataset for later use
# arguments:
#   - scenario: current scenario selected. 1: simple dataset, 2: correlated dataset, 3: weather dataset.
#   - t_v_t_split: train, validation test split, an array with 2 values indicating the proportion of train and valid.
#   - W: the size of the training window.
#   - device: where to place the data (cpu or gpu).
#   - signals: the actual input data.
#   - labels: labels for each signals if available.
def get_datasets(scenario, t_v_t_split, W, device, signals, labels=None):
    T = signals.shape[0]
    features = signals.shape[1]
    train_valid_time = math.floor(t_v_t_split[0] * T)
    valid_test_time = math.floor((t_v_t_split[0] + t_v_t_split[1]) * T)
    if (scenario == 0) or (scenario == 1):
        # normalize the signal
        normalized_signals = (signals - np.mean(signals, axis=0)) / np.std(signals, axis=0)
        # create train/valid/test split
        train_timeseries_signals = normalized_signals[0:train_valid_time]
        valid_timeseries_signals = normalized_signals[train_valid_time:valid_test_time]
        test_timeseries_signals = normalized_signals[valid_test_time:]

        train_timeseries_labels = labels[0:train_valid_time]
        valid_timeseries_labels = labels[train_valid_time:valid_test_time]
        test_timeseries_labels = labels[valid_test_time:]

        train_dataset = SyntheticDataset([train_timeseries_signals, train_timeseries_labels], features, window_size=W,
                                         device=device)
        valid_dataset = SyntheticDataset([valid_timeseries_signals, valid_timeseries_labels], features, device=device)
        test_dataset = SyntheticDataset([test_timeseries_signals, test_timeseries_labels], features, device=device)
    elif scenario == 2:
        features = 3
        normalized_signals = (signals - np.mean(signals, axis=0)) / np.std(signals, axis=0)

        train_dataset = RealisticDataset(normalized_signals[0:train_valid_time, :3], features, window_size=W,
                                         device=device)
        valid_dataset = RealisticDataset(normalized_signals[train_valid_time:valid_test_time, :3], features,
                                         window_size=W,
                                         device=device)
        test_dataset = RealisticDataset(normalized_signals[valid_test_time:, :3], features, window_size=W,
                                        device=device)
    elif scenario == 3:
        features = 2
        #        normalized_signals = (signals - np.mean(signals, axis=0)) / np.std(signals, axis=0)
        #normalized_signals = (signals-np.exp(np.mean(np.log(signals), axis = 0)))
        train_dataset = RealisticDataset(signals[0:train_valid_time, 3:], features, window_size=W, device=device)
        valid_dataset = RealisticDataset(signals[train_valid_time:valid_test_time, 3:], features, window_size=W,
                                         device=device)
        test_dataset = RealisticDataset(signals[valid_test_time:, 3:], features, window_size=W, device=device)
    else:
        train_dataset = None
        valid_dataset = None
        test_dataset = None

    return features, \
           train_dataset, \
           valid_dataset, \
           test_dataset
