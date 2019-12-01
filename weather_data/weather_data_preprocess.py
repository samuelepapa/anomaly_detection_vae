# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 21:49:11 2019

@author: Henrik
"""

# %% import libraries

import pandas as pd
import matplotlib as plt

# %% import data

# select only Los Angeles and Houston (lowest number of missing values)
df_humidity = pd.read_csv('weather_data/humidity.csv')[['datetime', 'Los Angeles', 'Houston']]
df_humidity.columns = ['datetime', 'humid_LA', 'humid_HO']

df_pressure = pd.read_csv('weather_data/pressure.csv')[['datetime', 'Los Angeles', 'Houston']]
df_pressure.columns = ['datetime', 'press_LA', 'press_HO']

df_temperature = pd.read_csv('weather_data/temperature.csv')[['datetime', 'Los Angeles', 'Houston']]
df_temperature.columns = ['datetime', 'temp_LA', 'temp_HO']

df_wind_dir = pd.read_csv('weather_data/wind_direction.csv')[['datetime', 'Los Angeles', 'Houston']]
df_wind_dir.columns = ['datetime', 'dir_LA', 'dir_HO']

df_wind_speed = pd.read_csv('weather_data/wind_speed.csv')[['datetime', 'Los Angeles', 'Houston']]
df_wind_speed.columns = ['datetime', 'speed_LA', 'speed_HO']

# %% combine variables

# Los Angeles dataframe
df_LA = pd.concat([df_humidity[['datetime', 'humid_LA']], df_pressure['press_LA']], axis=1)
df_LA = pd.concat([df_LA, df_temperature['temp_LA']], axis=1)
df_LA = pd.concat([df_LA, df_wind_dir['dir_LA']], axis=1)
df_LA = pd.concat([df_LA, df_wind_speed['speed_LA']], axis=1)

# houston dataframe
df_HO = pd.concat([df_humidity[['datetime', 'humid_HO']], df_pressure['press_HO']], axis=1)
df_HO = pd.concat([df_HO, df_temperature['temp_HO']], axis=1)
df_HO = pd.concat([df_HO, df_wind_dir['dir_HO']], axis=1)
df_HO = pd.concat([df_HO, df_wind_speed['speed_HO']], axis=1)

# %% check for NA values

df_LA.isna()
df_HO.isna()

# first row is NA in both, so just drop it
df_LA = df_LA.iloc[1:]
df_HO = df_HO.iloc[1:]

# %% imputation
# to do here:
# - impute missing values using simple univariate time series methods, e.g.
#    pandas' 'interpolate' function: 
#    https://stackoverflow.com/questions/49308530/missing-values-in-time-series-in-python
#    Note: a more proper way of doing this would be to fit a time series model and predict the missing values

# df_LA.index[df_LA['humid_LA'].isna()].tolist() # try experimenting with which interpolate funciton yields lowest MSE

# impute LA dataframe
df_LA['humid_LA'].interpolate(method='linear', inplace=True)
df_LA['press_LA'].interpolate(method='linear', inplace=True)
df_LA['temp_LA'].interpolate(method='linear', inplace=True)
df_LA['dir_LA'].interpolate(method='linear', inplace=True)
df_LA['speed_LA'].interpolate(method='linear', inplace=True)

# impute houston dataframe
df_HO['humid_HO'].interpolate(method='linear', inplace=True)
df_HO['press_HO'].interpolate(method='linear', inplace=True)
df_HO['temp_HO'].interpolate(method='linear', inplace=True)
df_HO['dir_HO'].interpolate(method='linear', inplace=True)
df_HO['speed_HO'].interpolate(method='linear', inplace=True)

# check missingness
df_LA.isna().any()
df_HO.isna().any()

# %% make ready for dataset

df_LA = df_LA.drop(['datetime'], axis=1)
df_HO = df_HO.drop(['datetime'], axis=1)

# %% correlations

df_LA.corr()

df_LA.to_csv("LA.csv", header=True, index=False)
df_HO.to_csv("HO.csv", header=True, index=False)

# %% to numpy array

# numpy array
df_LA = df_LA.to_numpy()
df_HO = df_HO.to_numpy()

# %% dataset
# get the data into a dataloader 
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
        return sample.to(self.device)


dataset_LA = RealisticDataset(df_LA, 5)
dataset_HO = RealisticDataset(df_HO, 5)

# %% dataloader

loader_LA = DataLoader(dataset_LA, batch_size=4, shuffle=False)
loader_HO = DataLoader(dataset_LA, batch_size=4, shuffle=False)
