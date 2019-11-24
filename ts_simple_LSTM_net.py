import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as distributions


# x[batch_size, num_features]
# z[batch_size, latent_features]
class Standard_LSTM(nn.Module):
    def __init__(self, input_dimension, param_size, hidden_dim):
        super(Standard_LSTM, self).__init__()

        self.input_dimension = input_dimension
        self.hidden_dim = hidden_dim
        self.param_size = param_size
        self.lstm = nn.LSTM(input_dimension, hidden_dim)
        self.hidden2params = nn.Linear(hidden_dim, param_size * input_dimension)

    def forward(self, x):
        outputs = {}
        time_length = len(x)
        outputs["x_input"] = x

        x = x.permute(1, 0, 2)
        #        x = x.view(time_length, self.batch_size, self.input_dimension)
        # lstm_out is the output of the last layer of hidden units [seq_len, batch, num_directions * hidden_size]
        # h is the hidden states at the last time step
        # c is the cell state at the last time step
        lstm_out, (h, c) = self.lstm(x)

        # linear wants [batch, seq_len, hidden_size]
        linear_in = lstm_out.permute(1, 0, 2)
        # take output of hidden layers at each time step h_t and run it through a fully connected layer
        params = self.hidden2params(linear_in)

        outputs["x_hat_params"] = params
        outputs["param_size"] = self.param_size
        return outputs


def loss_function_normal(model_output):
    # unpack the required quantities
    x_true = model_output["x_input"]

    input_dimension = x_true.shape[2]

    if model_output["x_hat_params"].shape[2] != 2 * input_dimension:
        raise ValueError("Wrong input dimensions or number of parameters in the output")

    mu, log_var = torch.chunk(model_output["x_hat_params"], 2, dim=2)
    sigma = torch.exp(log_var / 2)

    seq_length = mu.shape[1]
    # iterate over each time step in the sequence to compute NLL and KL terms

    t = 0
    cov_matrix = torch.diag_embed(sigma[:, t, :])
    # define the distribution
    p = distributions.MultivariateNormal(mu[:, t, :], cov_matrix)
    log_prob = p.log_prob(x_true[:, t + 1, :])

    for t in range(1, seq_length - 1):
        # print(t)
        # construct (diagonal) covariance matrix for each time step based on
        # the estimated var from the model
        cov_matrix = torch.diag_embed(sigma[:, t, :])

        # define the distribution
        p = distributions.MultivariateNormal(mu[:, t, :], cov_matrix)

        log_prob += p.log_prob(x_true[:, t + 1, :])
        # print(x_true.shape)

    NLL = - torch.mean(log_prob, dim=0) / seq_length
    return NLL
