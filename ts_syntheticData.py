import timesynth as ts  # install cmd: pip install git+https://github.com/TimeSynth/TimeSynth.git
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt


# %% data generation

class SyntheticDataset(Dataset):
    """Setup data for the time series data"""

    def __init__(self, data, dimensions, window_size=100, device='cpu', transform=None):
        # when window_size = 1 it means that it takes one time-step at a time
        self.device = device
        self.data = torch.tensor((data[0] - np.mean(data[0], axis=0)) / np.std(data[0], axis=0),
                                 dtype=torch.float).view(-1, dimensions)
        self.labels = torch.tensor(data[1], dtype=torch.float)
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
        label = self.labels[idx: idx + self.window_size]
        if self.transform:
            sample = self.transform(sample)
        return sample.to(self.device), label.to(self.device)

    def get_data(self):
        return self.data, self.labels

def generate_timeseries(signals,
                        T=100,  # number of samples taken from the time interval [0, 1000]
                        noise_std=0.01,  # standard deviation of the Gaussian Noise added to the signal
                        transforms=None,  # a list of anonymous functions e.g [lambda x: x**2, lambda x: np.sin(x)]
                        transforms_std=None  # if defined, should be e.g. [0.1,0.2,0.5] for 3 transforms
                        ):
    # used to define the time scale
    time_sampler = ts.TimeSampler(stop_time=1000)

    # create the time samples
    regular_time_samples = time_sampler.sample_regular_time(num_points=T)

    # define the standard gaussian white noise to add
    white_noise = ts.noise.GaussianNoise(std=noise_std)

    # the list of all the time serieses
    timeserieses = []
    for signal_type, params in signals:
        if signal_type == "sinusoid":  # sinusoidal signal
            sinusoid = ts.signals.Sinusoidal(**params)
            timeserieses.append(ts.TimeSeries(sinusoid, noise_generator=white_noise))
        if signal_type == "ar":  # autoregressive process
            ar_p = ts.signals.AutoRegressive(**params)
            timeserieses.append(ts.TimeSeries(signal_generator=ar_p))
        if signal_type == "gp":  # gaussian process
            gp = ts.signals.GaussianProcess(**params)
            timeserieses.append(ts.TimeSeries(gp, noise_generator=white_noise))

    # convert to numpy array and select only the signals
    output_samples = np.array([cur_timeseries.sample(regular_time_samples)[0]
                               for cur_timeseries in timeserieses])

    # apply transforms to obtain correlated time series. Each transformation will be
    # applied to each time series sequence. So only the original/untransformed sequences
    # will be truly uncorrelated
    if transforms is not None:
        # if no noise/standard deviation was given for the transformations just use the noise_std
        if transforms_std is None:
            transforms_std = [noise_std] * len(transforms)

        # apply the transforms
        output_samples_transforms = output_samples
        for f_idx, f in enumerate(transforms):
            std = transforms_std[f_idx]
            transformed_seq = f(output_samples) + np.random.randn(T) * std
            output_samples_transforms = np.append(output_samples_transforms,
                                                  transformed_seq,
                                                  axis=0)
        output_samples = output_samples_transforms

    # transpose to allow [time x features]
    return output_samples.T


# %% define anomaly function

def insert_anomalies(timeseries_samples, p=0.01, magnitude=1):
    import random as rnd
    timeseries_samples_with_anomalies = np.zeros(timeseries_samples.shape)
    anomaly = magnitude * np.ones(timeseries_samples.shape[1])
    labels = []
    t = -1
    for sample in timeseries_samples:
        t += 1
        if rnd.random() < p:
            labels.append(1)
            if rnd.random() < 0.5:
                timeseries_samples_with_anomalies[t] = sample + anomaly
            else:
                timeseries_samples_with_anomalies[t] = sample - anomaly

        else:
            labels.append(0)
            timeseries_samples_with_anomalies[t] = sample

    return timeseries_samples_with_anomalies, labels


# %% testing time series generation:
"""
# gaussian process with linear kernel
signals = [
    ("gp", {"kernel": 'Linear'})
]

# sine sequences
signals = [
    ("sinusoid", {"frequency": 1.25}),
    ("sinusoid", {"frequency": 1})
]

# create the timeseries
#train_timeseries = generate_timeseries(signals, T=1000, noise_std=0.1)
train_timeseries = generate_timeseries(signals, T=1000, noise_std=0.1,
                                       transforms=[lambda x: x + 10,
                                                   lambda x: x ** 2 + 5,
                                                   lambda x: np.sin(x) + 6],
                                       transforms_std=[0.7, 0.3, 0.4])
# plot them
plt.plot(train_timeseries)
train_timeseries.shape

# eyeball correlation matrix
import pandas as pd

data = pd.DataFrame(train_timeseries)
data.corr()
plt.matshow(data.corr())
plt.colorbar()


# end of testing
"""
