import copy

import numpy as np
from util import *

from neuralnet import *


def train(model, x_train, y_train, x_valid, y_valid, config):
    """
    TODO: Train your model here.
    Learns the weights (parameters) for our model
    Implements mini-batch SGD to train the model.
    Implements Early Stopping.
    Uses config to set parameters for training like learning rate, momentum, etc.

    args:
        model - an object of the NeuralNetwork class
        x_train - the train set examples
        y_train - the test set targets/labels
        x_valid - the validation set examples
        y_valid - the validation set targets/labels

    returns:
        the trained model
    """
    patience = 5  # for early stopping
    epochs = config['epochs']
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    momentum = config["momentum"]
    gamma = config['momentum_gamma']
    L2_penalty = config['L2_penalty']
    early_stop_epoch = config['early_stop_epoch']
    epochs_with_increase = 0  # check if epochs with increase is = to patience then early stop
    stopping_epoch = 0
    v_loss_previous = 0
    best_model = None
    flag_found_best_model = False
    # For 100 epochs...

    t_loss = []
    t_acc = []
    v_loss = []
    v_acc = []

    for epoch in range(0, epochs):
        # shuffle
        shuffle(x_train, y_train)
        print(f"epoch: {epoch}")
        # Generate minibatches
        train_minibatches = util.generate_minibatches((x_train, y_train), batch_size)
        early_stop = config["early_stop"]

        # Train the model over the minibatches
        minibatch_t_loss = []
        minibatch_t_acc = []

        for minibatch_t in train_minibatches:
            # first forward propagate
            _, _ = model(minibatch_t[0], minibatch_t[1])

            # backward propagate to find errors and update weights
            model.backward()

            # now measure loss and accuracy for minibatch training
            acc, loss = model(minibatch_t[0], minibatch_t[1])

            minibatch_t_loss = np.append(minibatch_t_loss, loss)
            minibatch_t_acc = np.append(minibatch_t_acc, acc)

        avg_t_epoch_loss = np.average(minibatch_t_loss)
        t_loss = np.append(t_loss, avg_t_epoch_loss)

        avg_t_epoch_acc = np.average(minibatch_t_acc)
        t_acc = np.append(t_acc, avg_t_epoch_acc)

        # find validation loss and accuracy for every epoch
        val_epoch_accuracy, val_epoch_loss = modelTest(model, x_valid, y_valid)
        v_loss = np.append(v_loss, val_epoch_loss)
        v_acc = np.append(v_acc, val_epoch_accuracy)

        if early_stop is True and epoch > early_stop_epoch and flag_found_best_model is False:

            if v_loss[epoch] > v_loss_previous:
                epochs_with_increase += 1
            else:
                epochs_with_increase = 0
                best_model = copy.deepcopy(model)
                print(stopping_epoch)
                # set best model now because if the next 5 epochs show higher val_loss that means
                # the model at this epoch must be the best and when epochs_with_increase hits 5

            v_loss_previous = v_loss[epoch]

            if epochs_with_increase == patience:
                print(f"Stopped at epoch: {epoch}")
                stopping_epoch = epoch - 5  # 5 epochs ago was our best model
                flag_found_best_model = True
                # best_model = copy.deepcopy(model)
            elif epoch == epochs - 1:  # last epoch
                stopping_epoch = epoch
                best_model = copy.deepcopy(model)

    plots(t_loss, t_acc, v_loss, v_acc, stopping_epoch)

    return best_model


# This is the test method
def modelTest(model, X_test, y_test):
    """
    TODO
    Calculates and returns the accuracy & loss on the test set.

    args:
        model - the trained model, an object of the NeuralNetwork class
        X_test - the test set examples
        y_test - the test set targets/labels

    returns:
        test accuracy
        test loss
    """

    test_accuracy, test_loss = model(X_test, y_test)
    return test_accuracy, test_loss

