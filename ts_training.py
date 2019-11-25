
import matplotlib.pyplot as plt
import numpy as np


def train_network(device, train_loader, valid_loader, epochs, net, loss_function, optimizer, scheduler=None,
                  plotting=True):
    train_loss = []
    train_KL = []
    print("Starting training")

    net.train()
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

            loss_params = loss_function(output_model, device)
            batch_loss.append(loss_params["loss"].item())
            if "KL" in loss_params.keys():
                batch_KL.append(loss_params["KL"].item())
            loss_params["loss"].backward()
            optimizer.step()

        train_loss.append(np.mean(batch_loss))
        train_KL.append(np.mean(batch_KL))
        if epoch % 10 == 0:
            ax1 = plt.subplot(1, 2, 1)
            ax1.plot(range(len(train_loss)), train_loss, label="Training loss")
            ax1.legend()
            ax2 = plt.subplot(1, 2, 2)
            ax2.plot(range(len(train_loss)), train_KL, label="Training KL")
            ax2.legend()
            plt.show()

    if (plotting):
        plt.plot(range(len(train_loss)), train_loss)
        plt.show()
    return net.state_dict()
