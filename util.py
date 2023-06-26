import copy
import os, gzip
import yaml
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import constants


def load_config(path):
    """
    Loads the config yaml from the specified path

    args:
        path - Complete path of the config yaml file to be loaded
    returns:
        yaml - yaml object containing the config file
    """
    return yaml.load(open(path, 'r'), Loader=yaml.SafeLoader)


def normalize_data(inp):
    """
    TODO DONE
    Normalizes inputs (on per channel basis of every image) here to have 0 mean and unit variance.
    This will require reshaping to separate the channels and then undoing it while returning

    args:
        inp : N X d 2D array where N is the number of examples and d is the number of dimensions

    returns:
        normalized inp: N X d 2D array

    """
    # reshape array to N  x 3 x 1024

    channels = 3
    channel_width = int(inp.shape[1] / channels)
    new_inp = inp.reshape(inp.shape[0], channels, channel_width)
    normalized = np.zeros(new_inp.shape)
    length = new_inp.shape[0]

    for i in range(length):
        for j in range(channels):
            normalized[i, j] = (new_inp[i][j] - np.mean(new_inp[i][j])) / np.std(new_inp[i][j])

    normalized = normalized.reshape(new_inp.shape[0], channels * channel_width)
    return normalized


def one_hot_encoding(labels, num_classes=10):
    """
    TODO DONE
    Encodes labels using one hot encoding.

    args:
        labels : N dimensional 1D array where N is the number of examples
        num_classes: Number of distinct labels that we have (10 for CIFAR-10)

    returns:
        oneHot : N X num_classes 2D array

    """
    labels_reshaped = labels.reshape((labels.shape[0]))
    one_hot = np.eye(num_classes)[labels_reshaped]

    return one_hot


def generate_minibatches(dataset, batch_size=64):
    """
        Generates minibatches of the dataset

        args:
            dataset : 2D Array N (examples) X d (dimensions)
            batch_size: mini batch size. Default value=64

        yields:
            (X,y) tuple of size=batch_size

        """

    X, y = dataset
    l_idx, r_idx = 0, batch_size
    while r_idx < len(X):
        yield X[l_idx:r_idx], y[l_idx:r_idx]
        l_idx, r_idx = r_idx, r_idx + batch_size

    yield X[l_idx:], y[l_idx:]


def calculateCorrect(y, t):  # Feel free to use this function to return accuracy instead of number of correct prediction
    """
    TODO CHECK DIMENSIONS OF Y (is it Nxd)
    Calculates the number of correct predictions

    args:
        y: Predicted Probabilities
        t: Labels in one hot encoding

    returns:
        the number of correct predictions // returning accuracy
    """
    total_hits = np.sum(np.argmax(y, axis=1) == np.argmax(t, axis=1))
    accuracy = (total_hits * 100) / y.shape[0]

    return accuracy


def append_bias(X):
    """
    TODO DONE
    Appends bias to the input
    args:
        X (N X d 2D Array)
    returns:
        X_bias (N X (d+1)) 2D Array
    """
    one = np.ones((X.shape[0], 1))
    # appended_x = np.append(X, one, 1)
    appended_x = np.insert(X, 0, 1, axis=1)
    return appended_x


def plots(trainEpochLoss, trainEpochAccuracy, valEpochLoss, valEpochAccuracy, earlyStop):
    """
    Helper function for creating the plots
    earlyStop is the epoch at which early stop occurred and will correspond to the best model. e.g.
    epoch=-1 means the last epoch was the best one
    """

    fig1, ax1 = plt.subplots(figsize=((24, 12)))
    epochs = np.arange(1, len(trainEpochLoss) + 1, 1)
    ax1.plot(epochs, trainEpochLoss, 'r', label="Training Loss")
    ax1.plot(epochs, valEpochLoss, 'g', label="Validation Loss")
    plt.scatter(epochs[earlyStop], valEpochLoss[earlyStop], marker='x', c='g', s=400, label='Early Stop Epoch')
    plt.xticks(ticks=np.arange(min(epochs), max(epochs) + 1, 10), fontsize=35)
    plt.yticks(fontsize=35)
    ax1.set_title('Loss Plots', fontsize=35.0)
    ax1.set_xlabel('Epochs', fontsize=35.0)
    ax1.set_ylabel('Cross Entropy Loss', fontsize=35.0)
    ax1.legend(loc="upper right", fontsize=35.0)
    plt.savefig(constants.saveLocation + "loss.png")
    plt.show()

    fig2, ax2 = plt.subplots(figsize=((24, 12)))
    ax2.plot(epochs, trainEpochAccuracy, 'r', label="Training Accuracy")
    ax2.plot(epochs, valEpochAccuracy, 'g', label="Validation Accuracy")
    plt.scatter(epochs[earlyStop], valEpochAccuracy[earlyStop], marker='x', c='g', s=400, label='Early Stop Epoch')
    plt.xticks(ticks=np.arange(min(epochs), max(epochs) + 1, 10), fontsize=35)
    plt.yticks(fontsize=35)
    ax2.set_title('Accuracy Plots', fontsize=35.0)
    ax2.set_xlabel('Epochs', fontsize=35.0)
    ax2.set_ylabel('Accuracy', fontsize=35.0)
    ax2.legend(loc="lower right", fontsize=35.0)
    plt.savefig(constants.saveLocation + "accuracy.png")
    plt.show()

    # Saving the losses and accuracies for further offline use
    pd.DataFrame(trainEpochLoss).to_csv(constants.saveLocation + "trainEpochLoss.csv")
    pd.DataFrame(valEpochLoss).to_csv(constants.saveLocation + "valEpochLoss.csv")
    pd.DataFrame(trainEpochAccuracy).to_csv(constants.saveLocation + "trainEpochAccuracy.csv")
    pd.DataFrame(valEpochAccuracy).to_csv(constants.saveLocation + "valEpochAccuracy.csv")


def shuffle(x_train, y_train):
    shuffle_index = np.random.permutation(len(x_train))  # get random permutation of sequence of numbers

    x_train = x_train[shuffle_index]  # shuffle data based on shuffleIndex
    y_train = y_train[shuffle_index]

    return x_train, y_train


def createTrainValSplit(x_train, y_train):
    """
    TODO -- DONE
    Creates the train-validation split (80-20 split for train-val). Please shuffle the data before creating the
    train-val split.
    """
    percentage = 0.2
    split = int(percentage * len(x_train))

    # first shuffle data
    shuffle_index = np.random.permutation(len(x_train))  # get random permutation of sequence of numbers

    x_train = x_train[shuffle_index]  # shuffle data based on shuffleIndex
    y_train = y_train[shuffle_index]

    val_images, val_labels = x_train[:split], y_train[:split]
    train_images, train_labels = x_train[split:], y_train[split:]

    return train_images, train_labels, val_images, val_labels


def load_data(path):
    """
    Loads, splits our dataset- CIFAR-10 into train, val and test sets and normalizes them

    args:
        path: Path to cifar-10 dataset
    returns:
        train_normalized_images, train_one_hot_labels, val_normalized_images, val_one_hot_labels,  test_normalized_images, test_one_hot_labels

    """

    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    cifar_path = os.path.join(path, constants.cifar10_directory)

    train_images = []
    train_labels = []
    val_images = []
    val_labels = []
    for i in range(1, constants.cifar10_trainBatchFiles + 1):
        images_dict = unpickle(os.path.join(cifar_path, f"data_batch_{i}"))
        data = images_dict[b'data']
        label = images_dict[b'labels']
        train_labels.extend(label)
        train_images.extend(data)
    train_images = np.array(train_images)
    train_labels = np.array(train_labels).reshape((len(train_labels), -1))
    train_images, train_labels, val_images, val_labels = createTrainValSplit(train_images, train_labels)

    train_normalized_images = normalize_data(train_images)  # TODO DONE
    train_one_hot_labels = one_hot_encoding(train_labels)  # TODO DONE

    val_normalized_images = normalize_data(val_images)  # TODO DONE
    val_one_hot_labels = one_hot_encoding(val_labels)  # TODO DONE

    test_images_dict = unpickle(os.path.join(cifar_path, f"test_batch"))
    test_data = test_images_dict[b'data']
    test_labels = test_images_dict[b'labels']
    test_images = np.array(test_data)
    test_labels = np.array(test_labels).reshape((len(test_labels), -1))
    test_normalized_images = normalize_data(test_images)  # TODO DONE
    test_one_hot_labels = one_hot_encoding(test_labels)  # TODO DONE
    # return train_normalized_images, train_labels, val_normalized_images, val_labels, test_normalized_images,
    # test_labels
    return train_normalized_images, train_one_hot_labels, val_normalized_images, val_one_hot_labels, test_normalized_images, test_one_hot_labels