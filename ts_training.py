import matplotlib.pyplot as plt
import numpy as np


def train_network(train_loader, valid_loader, epochs, net, loss_function, optimizer, scheduler=None, plotting=True):
    train_loss = []
    print("Starting training")

    net.train()
    for epoch in range(epochs):
        if plotting:
            if epoch % 10 == 0:
                print("Epoch: {}".format(epoch))
        batch_loss = []
        for batch_idx, x in enumerate(train_loader):
            # if (batch_idx % 10 == 0):
            #    print("Batch idx: ", batch_idx)
            net.zero_grad()
            input = x
            # targets = torch.from_numpy(train_timeseries_signals[1:]).float().unsqueeze(1)

            output_model = net(input)

            loss = loss_function(output_model)
            batch_loss.append(loss.item())

            loss.backward()
            optimizer.step()

        train_loss.append(np.mean(batch_loss))
        if epoch % 10 == 0:
            plt.close("all")
            plt.plot(range(len(train_loss)), train_loss)
            plt.show()

    if (plotting):
        plt.plot(range(len(train_loss)), train_loss)
        plt.show()
    return net.state_dict()
