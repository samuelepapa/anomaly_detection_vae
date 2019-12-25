# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 15:22:42 2019

"""
import math
import numpy as np
import torch
from torch import distributions, Tensor


# %% Outlier detection

# arguments
# - sequence: a tensor with dimensions T x D, where 
#      T is number of obs and D is feature dimensions
# - net: a trained model which outputs 2 params for each feature,
#      that is, a tensor of dimension T x 2*D
# - prob_threshold: a number between 0 and 1. An observation with 
#      probability < prob_threshold is labeled as an outlier
def detect_anomalies(sequence, net, device, prob_threshold, std=False, k=0):
    with torch.no_grad():
        net.eval()
        # get it to the device and put the batch dimension
        prepared_sequence = (sequence).to(device).unsqueeze(0)

        # run the model
        output_model = net(prepared_sequence, device)

        # get parameters of predicted data distribution for all time steps
        mu, logvar = torch.chunk(output_model["params"], 2, dim=-1)
        std_dev = torch.exp(logvar / 2)

        # drop batch dimension, if present (only needed for training)
        mu = mu.squeeze()
        std_dev = std_dev.squeeze()

        # main loop to measure outlier probability
        probs = []
        labels = [False] * mu.shape[0]
        for t in range(0, mu.shape[0] - 1):
            if std:
                # use the standard deviation to find if all the points are outside k times the std
                anomaly = False
                for feature in range(mu.shape[1]):
                    # if the signal is inside in just one feature it's not an anomaly
                    if (prepared_sequence[0, t + 1, feature] > mu[t, feature] + k * std_dev[t, feature]) or \
                            (prepared_sequence[0, t + 1, feature] < mu[t, feature] - k * std_dev[t, feature]):
                        anomaly = True
                        break
                if anomaly:
                    labels[t + 1] = True
            else:
                cov_matrix = torch.diag(std_dev[t, :])
                # define distribution with params estimated for time t+1
                # in the original sequence (that's simply t in the params
                # outputted by the model)
                p = distributions.MultivariateNormal(mu[t, :], cov_matrix)

                # measure the probability of the observation at time t+1
                # under the model and store the probability
                probability = torch.exp(p.log_prob(prepared_sequence[0, t + 1, :])).cpu().detach().numpy()
                probs.append(probability)

                # store outlier label
                if probability < prob_threshold:
                    labels[t + 1] = True

        # collect results in a dictionary
        outliers = {
            "outlier_label": labels,
            "probability": probs
        }
    return outliers


def detect_anomalies_VAE(sequence, net, device, prob_threshold):
    with torch.no_grad():
        net.eval()
        # get it to the device and put the batch dimension
        prepared_sequence = (sequence).to(device).unsqueeze(0)

        # run the model
        model_output = net(prepared_sequence, device)
        x_true = model_output["x_input"].permute(1, 0, 2)
        params = model_output["params"]
        mu, logvar = torch.chunk(params, 2, dim=-1)
        sigma = torch.exp(logvar / 2)

        z_mu = model_output["z_mu"]
        z_log_var = model_output["z_log_var"]

        seq_length = mu.shape[0]
        # iterate over each time step in the sequence to compute NLL and KL terms

        # dimensions [batch_size, dimension]
        ones_vector = torch.ones((z_mu.shape[1], z_mu.shape[2])).to(device)

        labels = [False] * mu.shape[0]
        probs = []
        for t in range(0, seq_length - 1):
            # print(t)
            # construct (diagonal) covariance matrix for each time step based on
            # the estimated var from the model
            cov_matrix = torch.diag_embed(sigma[t, :, :])

            # define the distribution
            p = distributions.Normal(mu[t, :, :], sigma[t, :, :])
            log_prob = torch.mean(p.log_prob(x_true[t + 1, :, :]), dim=-1)
            # KL-divergence
            kl = - 0.5 * torch.sum(ones_vector + z_log_var[t, :, :] - z_mu[t, :, :] ** 2 -
                                   torch.exp(z_log_var[t, :, :]))
            lower_bound_probability = torch.exp(log_prob - kl).cpu().detach().numpy()

            if lower_bound_probability < prob_threshold:
                labels[t + 1] = True
            probs.append(lower_bound_probability)

        # collect results in a dictionary
        outliers = {
            "outlier_label": labels,
            "probability": probs
        }
    return outliers

# %% test outlier detection

## select the sequence to test the network on
# sequence = valid_dataset.get_data()[0]
# prob_threshold = 0.1
#
# foo = detect_anomalies(sequence,net,0.0001)
# plt.plot(foo["probability"])
