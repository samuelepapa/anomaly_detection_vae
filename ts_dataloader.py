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
            ("sinusoid", {"frequency": 0.019}),
            ("sinusoid", {"frequency": 0.05}),
            ("sinusoid", {"frequency": 0.03}),
            ("sinusoid", {"frequency": 0.023}),
            ("sinusoid", {"frequency": 0.1}),
            ("sinusoid", {"frequency": 0.13}),
            ("sinusoid", {"frequency": 0.87}),
            ("sinusoid", {"frequency": 0.007}),
            ("sinusoid", {"frequency": 0.3})
        ]
        features = len(signals)

        # generate the time series using signals
        timeseries_signals = generate_timeseries(signals, T=T)
        timeseries_signals, timeseries_labels = insert_anomalies(timeseries_signals, magnitude=0.1)
        print(timeseries_signals.shape)
        # create train/valid/test split
        maximum = np.max(timeseries_signals)
        train_timeseries_signals = timeseries_signals[0:train_valid_time] / maximum
        valid_timeseries_signals = timeseries_signals[train_valid_time:valid_test_time] / maximum
        test_timeseries_signals = timeseries_signals[valid_test_time:] / maximum

        train_timeseries_labels = timeseries_labels[0:train_valid_time]
        valid_timeseries_labels = timeseries_labels[train_valid_time:valid_test_time]
        test_timeseries_labels = timeseries_labels[valid_test_time:]

        train_dataset = SyntheticDataset([train_timeseries_signals, train_timeseries_labels], features, window_size=W,
                                         device=device)
        valid_dataset = SyntheticDataset([valid_timeseries_signals, valid_timeseries_labels], features, window_size=W,
                                         device=device)
        test_dataset = SyntheticDataset([test_timeseries_signals, test_timeseries_labels], features, window_size=W,
                                        device=device)

        print("Dataset created.")
    elif dataset_id == 1:
        # sine sequences
        signals = [
            # ("sinusoid", {"frequency": 1.25}),
            ("sinusoid", {"frequency": 1}),
            # ("gp", {"kernel": "Linear", "c":1,"offset":-10, "variance": 0.1})
        ]


        # create the timeseries
        timeseries_signals = generate_timeseries(signals, T=T, noise_std=0.001,
                                                 transforms=[lambda x: x ** 3 + 10,
                                                             lambda x: x ** 2,
                                                             lambda x: np.sin(x)],
                                                 transforms_std=[0.007, 0.003, 0.004])
        # timeseries_signals = generate_timeseries(signals, T=T, noise_std=0.001,
        #                                         transforms=[lambda x: x,
        #                                                     lambda x: x,
        #                                                     lambda x: x],
        #                                         transforms_std=[0.01, 0.01, 0.01])
        # timeseries_signals = np.ones((T,5))*5
        #        timeseries_signals = generate_timeseries(signals, T=T, noise_std=0.001)
        # timeseries_signals = ((np.array(list(range(T)))-T/2)).reshape(T,1)
        features = timeseries_signals.shape[1]
        timeseries_signals, timeseries_labels = insert_anomalies(timeseries_signals, magnitude=0)
        print(timeseries_signals.shape)
        maximum = np.max(np.abs(timeseries_signals))

        train_timeseries_signals = timeseries_signals[0:train_valid_time] / maximum
        valid_timeseries_signals = timeseries_signals[train_valid_time:valid_test_time] / maximum
        test_timeseries_signals = timeseries_signals[valid_test_time:] / maximum

        import matplotlib.pyplot as plt
        plt.plot(list(range(len(train_timeseries_signals))), train_timeseries_signals)
        plt.show()

        train_timeseries_labels = timeseries_labels[0:train_valid_time]
        valid_timeseries_labels = timeseries_labels[train_valid_time:valid_test_time]
        test_timeseries_labels = timeseries_labels[valid_test_time:]

        train_dataset = SyntheticDataset([train_timeseries_signals, train_timeseries_labels], features, window_size=W,
                                         device=device)
        valid_dataset = SyntheticDataset([valid_timeseries_signals, valid_timeseries_labels], features, window_size=W,
                                         device=device)
        test_dataset = SyntheticDataset([test_timeseries_signals, test_timeseries_labels], features, window_size=W,
                                        device=device)

        print("Dataset created.")
    # set up the DataLoader
    # This dataloader will load data with shape [batch_size, time_length, features]
    return features, \
           train_dataset, \
           valid_dataset, \
           test_dataset
