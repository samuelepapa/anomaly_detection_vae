"""
Time Series Variational Inference LSTM

@author: Samuele Papa, Henrik Hviid Hansen
"""
import torch
from torch import nn, distributions


class Variational_LSTM(nn.Module):
    def __init__(self, input_dim, param_dist, hidden_dim_rec, hidden_dim_gen, latent_dim):
        super(Variational_LSTM, self).__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.hidden_dim_rec = hidden_dim_rec
        self.hidden_dim_gen = hidden_dim_gen

        self.recognition_LSTM = nn.LSTM(input_size=input_dim,
                                        hidden_size=hidden_dim_rec,  # 2 because we want mu and sigma
                                        num_layers=1
                                        )
        self.rec2latent = nn.Linear(hidden_dim_rec, 2 * latent_dim)

        self.generator_LSTM = nn.LSTM(input_size=input_dim + latent_dim,
                                      hidden_size=hidden_dim_gen,  # 2* because we want parameters of a normal dist
                                      num_layers=1
                                      )

        self.gen2features = nn.Linear(hidden_dim_gen, param_dist * input_dim)

    def forward(self, x, device):
        outputs = {}
        outputs["x_input"] = x
        batch_size = x.shape[0]
        # --- recognition LSTM to generate latent variable(s) for each time step --- #

        # get x in the right format: [seq_length, batch_size, input_dimension]
        x_latent = x.permute(1, 0, 2)
        seq_length = x_latent.shape[0]
        lstm_out, (h, c) = self.recognition_LSTM(x_latent)
        # linear wants [batch, seq_len, hidden_size]
        rec_linear_in = lstm_out.permute(1, 0, 2)
        x_latent = self.rec2latent(rec_linear_in)  # outputs the latent variable parameters mu and sigma

        # make the network learn the log of the variance it is non-negative (log only takes pos x)
        mu, log_var = torch.chunk(x_latent, 2, dim=-1)
        sigma = torch.exp(log_var / 2)  # std of the latent variable distribution
        # reparameterization trick happens here:
        epsilon = 0
        with torch.no_grad():
            epsilon = torch.randn(batch_size,
                                  seq_length,
                                  self.latent_dim,
                                  # latent_samples here, but should only be one as we concatenate it to the oringinal x
                                  ).to(device)  # 10 is the num of latent samples
        # create the random latent variable

        z = mu + sigma * epsilon

        # store intermediate results and latent variables
        outputs["z"] = z
        outputs["z_mu"] = mu
        outputs["z_sigma"] = sigma
        outputs["z_log_var"] = log_var

        # --- ordinary LSTM to predict the next x in the time sequence --- #
        # concatenate the latent variables with the original input x
        # [sequence_len, batch_size, dimensions]
        x = x.permute(1, 0, 2)
        z = z.permute(1, 0, 2)
        x_aug = torch.cat((x, z), dim=-1)  # aug = augmented x, because we augment x with z
        # x_aug = x_aug.view(seq_len, self.batch_size, self.input_dim + self.latent_dim)

        # run it through the ordinary LSTM
        lstm_out, (h, c) = self.generator_LSTM(x_aug)
        # [batch_size, sequence_len, dimensions]
        gen_linear_in = lstm_out.permute(1, 0, 2)
        params = self.gen2features(gen_linear_in)

        # store the outputs in the form (batch_size, seq_length, input_dim)
        outputs["params"] = params

        return outputs


def loss_normal2d(model_output, device):
    # unpack the required quantities
    x_true = model_output["x_input"]
    params = model_output["params"]
    mu, logvar = torch.chunk(params, 2, dim=-1)
    sigma = torch.exp(logvar / 2)

    z_mu = model_output["z_mu"]
    z_log_var = model_output["z_log_var"]

    seq_length = mu.shape[1]
    # iterate over each time step in the sequence to compute NLL and KL terms

    t = 0
    cov_matrix = torch.diag_embed(sigma[:, t, :])
    # define the distribution
    p = distributions.MultivariateNormal(mu[:, t, :], cov_matrix)
    log_prob = p.log_prob(x_true[:, t + 1, :])
    # dimensions [batch_size, dimension]
    ones_vector = torch.ones((z_mu.shape[0], z_mu.shape[2])).to(device)
    kl = 0.5 * torch.sum(ones_vector + z_log_var[:, t, :] - z_mu[:, t, :] ** 2 - torch.exp(z_log_var[:, t, :]), dim=-1)

    for t in range(1, seq_length - 1):
        # print(t)
        # construct (diagonal) covariance matrix for each time step based on
        # the estimated var from the model
        cov_matrix = torch.diag_embed(sigma[:, t, :])

        # define the distribution
        p = distributions.MultivariateNormal(mu[:, t, :], cov_matrix)

        log_prob += p.log_prob(x_true[:, t + 1, :])
        kl += 0.5 * torch.sum(ones_vector + z_log_var[:, t, :] - z_mu[:, t, :] ** 2 -
                              torch.exp(z_log_var[:, t, :]))

    NLL, KL = -torch.mean(log_prob, dim=0) / (seq_length - 1), -torch.mean(kl, dim=0) / (seq_length - 1)
    ELBO = NLL + KL

    return {
        "loss": ELBO,
        "ELBO": ELBO,
        "NLL": NLL,
        "KL": KL
    }
