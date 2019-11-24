from ts_syntheticData import generate_timeseries, insert_anomalies, SyntheticDataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import math


def get_train_valid_test_signals(T, W, dataset_id, t_v_t_split, device):
    ADDANOMALIES = False
    train_valid_time = math.floor(t_v_t_split[0] * T)
    valid_test_time = math.floor((t_v_t_split[0] + t_v_t_split[1]) * T)
    if dataset_id == 0:
        signals = [
            ("sinusoid", {"frequency": 1.25}),
            ("sinusoid", {"frequency": 1.5}),
            ("sinusoid", {"frequency": 1.3})
        ]
        features = len(signals)

        # generate the time series using signals
        timeseries_signals = generate_timeseries(signals, T=T)
        if ADDANOMALIES:
            timeseries_signals, timeseries_labels = insert_anomalies(timeseries_signals, magnitude=0)

        # create train/valid/test split
        train_timeseries_signals = timeseries_signals[0:train_valid_time]
        valid_timeseries_signals = timeseries_signals[train_valid_time:valid_test_time]
        test_timeseries_signals = timeseries_signals[valid_test_time:]

        train_dataset = SyntheticDataset(train_timeseries_signals, features, window_size=W, device=device)
        valid_dataset = SyntheticDataset(valid_timeseries_signals, features, window_size=W, device=device)
        test_dataset = SyntheticDataset(test_timeseries_signals, features, window_size=W, device=device)

        print("Dataset created.")
    elif dataset_id == 1:
        # sine sequences
        signals = [
            ("sinusoid", {"frequency": 1.25}),
            ("sinusoid", {"frequency": 1})
        ]

        # create the timeseries
        timeseries_signals = generate_timeseries(signals, T=1000, noise_std=0.1,
                                                 transforms=[lambda x: x + 10,
                                                             lambda x: x ** 2 + 5,
                                                             lambda x: np.sin(x) + 6],
                                                 transforms_std=[0.7, 0.3, 0.4])
        if ADDANOMALIES:
            timeseries_signals, train_timeseries_labels = insert_anomalies(timeseries_signals, magnitude=0)

        features = len(signals)
        # create the Synthetic Dataset
        train_dataset = SyntheticDataset(timeseries_signals, features, window_size=W, device=device)

        print("Dataset created.")
    # set up the DataLoader
    # This dataloader will load data with shape [batch_size, time_length, features]
    return features, \
           train_dataset, \
           valid_dataset, \
           test_dataset
