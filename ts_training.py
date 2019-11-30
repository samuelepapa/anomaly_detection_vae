import matplotlib.pyplot as plt
import numpy as np
import torch
from ts_anomaly_function import detect_anomalies


def anomaly_detection_accuracy(ground_truth, predictions):
    correct = 0
    total = len(predictions) - 1
    false_positives = 0
    false_negatives = 0
    for i in range(total):
        if predictions[i] == ground_truth[i]:
            correct += 1
        elif predictions[i] == False:
            false_negatives += 1
        elif predictions[i] == True:
            false_positives += 1
    return {
        "correct": correct,  # number of labels correctly predicted
        "false_positives": false_positives,
        # false positives (the datapoint was not an anomaly but it was predicted as one)
        "false_negatives": false_negatives,
        # false negatives (the datapoint was an anomaly abut it was not predicted as one)
        "total": total  # total number of datapoints in the sequence
    }


def train_network(device, train_loader, valid_dataset, epochs, net, loss_function, optimizer, beta_annealing=None,
                  scheduler=None, plotting=True, p_anomaly=0.01):
    train_loss = []
    train_KL = []
    print("Training has started.")

    if beta_annealing is None:
        beta = 1
    else:
        beta = beta_annealing(0, 0)

    # setup validation sequence
    valid_sequence = valid_dataset.get_data()[0]
    # insert batch dimension
    valid_sequence_batched = valid_sequence.to(device).unsqueeze(0)
    # get the labels (if they are not available they will be set to None
    valid_labels = valid_dataset.get_data()[1].to(device)
    # validation loss
    valid_loss = []
    # accuracy of the anomaly detector
    valid_accuracy = {"f_positives": np.array([]),
                      "f_negatives": np.array([]),
                      "correct": np.array([]),
                      "total": np.array([])}
    train_accuracy = {

    }
    # cycle through the epochs
    for epoch in range(epochs):
        if plotting:
            if epoch % 10 == 0:
                print("Epoch: {}".format(epoch))

        batch_loss = []
        batch_KL = []

        # put in training mode
        net.train()

        for batch_idx, x in enumerate(train_loader):
            net.zero_grad()

            # shape: [batch_size, seq_len, features]
            input = x[0]  # x[0] is the data x[1] is the labels

            # FORWARD PASS
            output_model = net(input, device)

            # check if beta annealing is happening
            if beta_annealing is None:
                loss_params = loss_function(output_model, device)
            else:
                loss_params = loss_function(output_model, device, beta)

            # add current loss to the batch loss
            batch_loss.append(loss_params["loss"].item())
            # if there is a KL divergence, add it
            if "KL" in loss_params.keys():
                batch_KL.append(loss_params["KL"].item())

            # BACKPROPAGATION
            loss_params["loss"].backward()
            optimizer.step()

        # add to the training loss the average of the batch loss
        train_loss.append(np.mean(batch_loss))
        # only if there has been a KL, then add its average to the train_KL
        if len(batch_KL) > 0:
            train_KL.append(np.mean(batch_KL))
        # every 10 epochs
        if epoch % 10 == 0:
            net.eval()
            with torch.no_grad():
                # forward pass using the validation sequence
                output_model = net(valid_sequence_batched, device)
                # check if beta_annealing is happening
                if beta_annealing is None:
                    loss_params = loss_function(output_model, device)
                else:
                    loss_params = loss_function(output_model, device, beta)
                # add to the validation loss
                valid_loss.append(loss_params["loss"].cpu().item())

                # if I am using the labelled validation set
                if valid_dataset.has_labels():
                    # detect the anomalies
                    anomaly_data = detect_anomalies(valid_sequence, net, device, p_anomaly)
                    # what we predict is an anomaly
                    predictions = anomaly_data["outlier_label"]
                    # find the accuracy of the prediction
                    anomaly_accuracy = anomaly_detection_accuracy(valid_labels[1:], predictions[:-1])

                    valid_accuracy["total"] = np.append(valid_accuracy["total"], [anomaly_accuracy["total"]])
                    valid_accuracy["correct"] = np.append(valid_accuracy["correct"], [anomaly_accuracy["correct"]])
                    valid_accuracy["f_positives"] = np.append(valid_accuracy["f_positives"],
                                                              [anomaly_accuracy["false_positives"]])
                    valid_accuracy["f_negatives"] = np.append(valid_accuracy["f_negatives"],
                                                              [anomaly_accuracy["false_negatives"]])
            if plotting:
                plt.figure(figsize=(15, 5))
                if len(train_KL) == 0:

                    ax1 = plt.subplot(1, 3, 1)
                    ax1.plot(range(len(train_loss)), train_loss, label="Training loss")
                    ax1.legend()
                    ax2 = plt.subplot(1, 3, 2)
                    ax2.plot(range(len(valid_accuracy["total"])), valid_accuracy["correct"] / valid_accuracy["total"],
                             label="Correctly labelled")
                    ax2.legend()
                    ax3 = plt.subplot(1, 3, 3)
                    ax3.plot(range(len(valid_loss)), valid_loss, label="Validation loss")
                    ax3.legend()
                    plt.show()

                else:
                    ax1 = plt.subplot(2, 2, 1)
                    ax1.plot(range(len(train_loss)), train_loss, label="Training loss")
                    ax1.legend()
                    ax2 = plt.subplot(2, 2, 2)
                    ax2.plot(range(len(train_loss)), train_KL, label="Training KL")
                    ax2.legend()
                    ax3 = plt.subplot(2, 2, 3)
                    ax3.plot(range(len(valid_accuracy["total"])), valid_accuracy["correct"] / valid_accuracy["total"],
                             label="Correctly labelled")
                    ax3.legend()
                    ax4 = plt.subplot(2, 2, 4)
                    ax4.plot(range(len(valid_loss)), valid_loss, label="Validation loss")
                    ax4.legend()
                    plt.show()

        # step in the scheduler and in the annealing of beta
        if scheduler != None:
            scheduler.step()
        if beta_annealing != None:
            beta = beta_annealing(beta, epoch)
    if (plotting):
        plt.plot(range(len(train_loss)), train_loss)
        plt.show()
    return net.state_dict()
