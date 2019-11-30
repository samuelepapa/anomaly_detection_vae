import matplotlib.pyplot as plt
import numpy as np


def train_network(device, train_loader, valid_loader, epochs, net, loss_function, optimizer, beta_annealing=None,
                  scheduler=None,
                  plotting=True):
    train_loss = []
    train_KL = []
    print("Starting training")

    net.train()
    if beta_annealing != None:
        beta = beta_annealing(0, 0)
    else:
        beta = 1

    for epoch in range(epochs):
        if plotting:
            if epoch % 10 == 0:
                print("Epoch: {}".format(epoch))

        batch_loss = []
        batch_KL = []

        for batch_idx, x in enumerate(train_loader):
            net.zero_grad()
            # shape: [batch_size, seq_len, features]
            input = x[0]

            output_model = net(input, device)
            if beta_annealing == None:
                loss_params = loss_function(output_model, device)
            else:

                loss_params = loss_function(output_model, device, beta)
            batch_loss.append(loss_params["loss"].item())
            if "KL" in loss_params.keys():
                batch_KL.append(loss_params["KL"].item())
            loss_params["loss"].backward()
            optimizer.step()

        train_loss.append(np.mean(batch_loss))
        if len(batch_KL) > 0:
            train_KL.append(np.mean(batch_KL))
        if epoch % 10 == 0:
            if len(train_KL) == 0:
                plt.plot(range(len(train_loss)), train_loss, label="Training loss")
                plt.legend()
                plt.show()
            else:
                ax1 = plt.subplot(1, 2, 1)
                ax1.plot(range(len(train_loss)), train_loss, label="Training loss")
                ax1.legend()
                ax2 = plt.subplot(1, 2, 2)
                ax2.plot(range(len(train_loss)), train_KL, label="Training KL")
                ax2.legend()
                plt.show()
        if scheduler != None:
            scheduler.step()
        if beta_annealing != None:
            beta = beta_annealing(beta, epoch)
    if (plotting):
        plt.plot(range(len(train_loss)), train_loss)
        plt.show()
    return net.state_dict()
