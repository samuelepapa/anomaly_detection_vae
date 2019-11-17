# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 21:14:16 2019

@author: Henrik
"""

# data handling
import os
import pickle as pkl
from urllib.request import urlretrieve

# torch and numpy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch import distributions
from torch import optim

# plotting
import matplotlib.pyplot as plt

# timesynth package for synthetic time series data creation: https://github.com/TimeSynth/TimeSynth
import timesynth as ts  # install cmd: pip install git+https://github.com/TimeSynth/TimeSynth.git

# %% set device

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

# %% data generation
dimensions = 1
T = 100000

# reference https://github.com/TimeSynth/TimeSynth/blob/master/TimeSynthExamples.ipynb
time_sampler = ts.TimeSampler(stop_time=1000)

irregular_time_samples = time_sampler.sample_irregular_time(num_points=T, keep_percentage=50)
regular_time_samples = time_sampler.sample_regular_time(num_points=T)
sinusoid = ts.signals.Sinusoidal(frequency=0.25)
white_noise = ts.noise.GaussianNoise(std=0.3)

timeseries = ts.TimeSeries(sinusoid, noise_generator=white_noise)

samples = timeseries.sample(regular_time_samples)[0]
samples = [[x] for x in samples]
# samples should be [samples, dimensions]

# continuous autoregressive
car = ts.signals.CAR(ar_param=0.9, sigma=0.01)
car_series = ts.TimeSeries(signal_generator=car)


# samples = car_series.sample(regular_time_samples)[0]


def generate_timeseries(signal_types, T=10000, noise_std=0.01):
    time_sampler = ts.TimeSampler(stop_time=1000)
    regular_time_samples = time_sampler.sample_regular_time(num_points=T)
    white_noise = ts.noise.GaussianNoise(std=noise_std)

    timeserieses = []
    for signal_type in signal_types:
        if signal_type == "sinusoid":
            timeserieses.append(ts.TimeSeries(sinusoid, noise_generator=white_noise))

    output_samples = np.array([cur_timeseries.sample(regular_time_samples)[0] for cur_timeseries in timeserieses])
    return output_samples.T


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


class SyntheticDataset(Dataset):
    """Binary MNIST dataset"""

    def __init__(self, data, dimensions, device='cpu', transform=None):
        self.device = device
        self.data = torch.tensor(data, dtype=torch.float).view(-1, dimensions)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample.to(self.device)


# %% make some data

batch_size = 1
dimensions = 4
train_timeseries = (generate_timeseries(["sinusoid", "sinusoid"]))
train_labeled_anomalies, labels = insert_anomalies(train_timeseries)
train_dataset = SyntheticDataset(train_labeled_anomalies, dimensions)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)


# %% model


class Variational_LSTM(nn.Module):
    def __init__(self, input_dim, latent_dim, batch_size):
        super(Variational_LSTM, self).__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.batch_size = batch_size

        self.recognition_LSTM = nn.LSTM(input_size=input_dim,
                                        hidden_size=2 * latent_dim,  # 2 because we want mu and sigma
                                        num_layers=1
                                        ).to(device)

        self.LSTM = nn.LSTM(input_size=input_dim + latent_dim,
                            hidden_size=2 * input_dim,  # 2* because we want parameters of a normal dist
                            num_layers=1
                            ).to(device)

    def forward(self, x):
        x = x.to(device)
        outputs = {}
        outputs["x_input"] = x

        # --- recognition LSTM to generate latent variable(s) for each time step --- #

        # get x in the right format: (seq_length, batch_size, input_dimension)
        x_latent = x.view(-1, self.batch_size, self.input_dim)
        seq_length = x_latent.shape[0]
        x_latent, _ = self.recognition_LSTM(x_latent)  # outputs the latent variable parameters mu and sigma

        # make the network learn the log of the variance it is non-negative (log only takes pos x)
        mu, log_var = torch.chunk(x_latent, 2, dim=-1)
        sigma = torch.exp(log_var / 2)  # std of the latent variable distribution

        # reparameterization trick happens here:
        epsilon = 0
        with torch.no_grad():
            epsilon = torch.randn(seq_length,
                                  self.latent_dim,
                                  # latent_samples here, but should only be one as we concatenate it to the oringinal x
                                  ).to(device)  # 10 is the num of latent samples

        # create the random latent variable
        z = mu.view(-1, self.latent_dim) + sigma.view(-1, self.latent_dim) * epsilon

        # store intermediate results and latent variables
        outputs["z"] = z
        outputs["z_mu"] = mu
        outputs["z_sigma"] = sigma
        outputs["z_log_var"] = log_var

        # --- ordinary LSTM to predict the next x in the time sequence --- #
        # concatenate the latent variables with the original input x
        x = x.view(-1, self.input_dim)
        x_aug = torch.cat((x, z), dim=-1)  # aug = augmented x, because we augment x with z
        x_aug = x_aug.view(-1, self.batch_size, self.input_dim + self.latent_dim)

        # run it through the ordinary LSTM
        x_hat, _ = self.LSTM(x_aug)
        x_hat_mu, x_hat_logvar = torch.chunk(x_hat, 2, dim=-1)
        x_hat_sigma = torch.exp(x_hat_logvar / 2)

        # store the outputs in the form (batch_size, seq_length, input_dim)
        outputs["x_hat_mu"] = x_hat_mu.view(-1, seq_length, self.input_dim)
        outputs["x_hat_sigma"] = x_hat_sigma.view(-1, seq_length, self.input_dim)

        return outputs


# fix seed for reproducibility
torch.manual_seed(42)

test_x = train_loader.dataset[0]
test_xp1 = train_loader.dataset[1]
test_xs = train_loader.dataset[0:5]
# print(test_x, "\n", test_xs)
net = Variational_LSTM(input_dim=dimensions,
                       latent_dim=2,
                       batch_size=batch_size)
lstm = net(test_x)
lstm = net(test_xs)

# %% loss function


model_output = lstm


def loss_normal2d(model_output):
    # unpack the required quantities
    x_true = model_output["x_input"]
    x_hat_mu = model_output["x_hat_mu"]
    x_hat_sigma = model_output["x_hat_sigma"]

    z_mu = model_output["z_mu"]
    z_log_var = model_output["z_log_var"]

    seq_length = x_hat_mu.shape[1]
    # iterate over each time step in the sequence to compute NLL and KL terms

    t = 0
    cov_matrix = torch.diag_embed(x_hat_sigma[0, t, :])
    # define the distribution
    p = distributions.MultivariateNormal(x_hat_mu[0, t, :], cov_matrix)
    log_prob = p.log_prob(x_true[t + 1, :])
    kl = -0.5 * torch.sum(1 + z_log_var[t, :, :] - z_mu[t, :, :] ** 2 -
                          torch.exp(z_log_var[t, :, :]))

    for t in range(1, seq_length - 1):
        print(t)
        # construct (diagonal) covariance matrix for each time step based on
        # the estimated var from the model
        cov_matrix = torch.diag_embed(x_hat_sigma[0, t, :])

        # define the distribution
        p = distributions.MultivariateNormal(x_hat_mu[0, t, :], cov_matrix)

        log_prob += p.log_prob(x_true[t + 1, :])
        kl += -0.5 * torch.sum(1 + z_log_var[t, :, :] - z_mu[t, :, :] ** 2 -
                               torch.exp(z_log_var[t, :, :]))

    NLL, KL = log_prob / seq_length - 1, kl / seq_length - 1
    ELBO = NLL - KL

    return -ELBO, -NLL, -KL


loss_normal2d(model_output)

# def loss_normal2d(mu_vector, sigma_vector, x_true):
#      # construct (diagonal) covariance matrix for each time step based on the estimated var from the model
#      cov_matrix = torch.diag_embed(sigma_vector)
#
#      # define the distribution
#      p = distributions.MultivariateNormal(mu_vector,cov_matrix)
#
#      # mean log likelihood over the time steps
#      log_prob = torch.mean(p.log_prob(x_true))
#
#      # return the NEGATIVE log likelihood which is to be minimized
#      return -log_prob
#

# loss_normal2d(mu_vector,sigma_vector,test_xs.to(device))

# %% training parameters
# epochs
num_epochs = 400
batch_size = 100  # note: corresponds to sequence length aka time steps here

input_dimensions = 2
train_timeseries = generate_timeseries(["sinusoid", "sinusoid"])
train_labeled_anomalies, labels = insert_anomalies(train_timeseries)
print(train_labeled_anomalies.shape)
train_dataset = SyntheticDataset(train_labeled_anomalies, input_dimensions)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

# init loss vector
train_loss, valid_loss = [], []

# initialize the network
net = Variational_LSTM(input_dim=input_dimensions,
                       latent_dim=1,
                       batch_size=batch_size).to(device)

# The Adam optimizer works really well with VAEs.
optimizer = optim.Adam(net.parameters(), lr=0.01)
loss_function = loss_normal2d

# %% training loop

# training loop
for epoch in range(num_epochs):

    print("Epoch: ", epoch)
    batch_loss = []

    # activate training mode
    net.train()

    # iterate over the batches
    for batch_idx, x in enumerate(train_loader):
        x = x.to(device)
        print(train_loader)
        print(batch_idx)
        print(x.shape)
        if batch_idx == 0:
            break

        # run the forward pass and store estimated mean and variances for each time step
        #            _,_,_,_, x_hat_mu, x_hat_sigma = net(x).values()
        model_output = net(x)

        # compute loss
        ELBO, NLL, KL = loss_function(model_output)
        #            print(ELBO.item(), NLL.item(), KL.item())

        # optimize
        optimizer.zero_grad()
        ELBO.backward()
        optimizer.step()

        # loss on this batch
        batch_loss.append(ELBO.item())

    # store the training losses for this epoch - this completes one iteration over the
    # whole dataset
    print("Training loss: ", np.round(np.mean(batch_loss), 4))
    train_loss.append(np.mean(batch_loss))

    #      # evaluate performance on validation set
    #      with torch.no_grad():
    #            # put model in evaluation mode
    #            net.eval()
    #
    #            # why only load a single batch here??
    #            x = next(iter(valid_loader)).to(device)
    #
    #            # run the forward pass and store reconstruction, latent z, mu and sd
    #            x_hat, z, mu, log_var = net(x).values()
    #
    #            # compute loss
    #            elbo, ll, kl = loss_function(x_hat, x, mu, log_var)
    #
    #            # store z for visualizing
    #            z = z.detach().to("cpu").numpy()
    #
    #            # store validation loss
    #            print("Validation loss: ", elbo.item())
    #            valid_loss.append(elbo.item())
    #            valid_ll.append(ll.item())
    #            valid_kl.append(kl.item())

    # dont start plotting till after epoch 2 due to high initial loss messing up the plot
    if epoch < 2:
        continue

    # plot training and validation loss
    x_range = np.arange(epoch + 1)
    plt.subplot(111)
    plt.plot(x_range[2:], train_loss[2:], 'red', label="train")
    #      plt.plot(x_range[2:],valid_loss[2:],'blue',label="valid")
    plt.ylabel("ELBO")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()
