import numpy as np
from neuralnet import Neuralnetwork
import copy


def check_grad(model, x_train, y_train):
    """
    TODO
        Checks if gradients computed numerically are within O(epsilon**2)

        args:
            model
            x_train: Small subset of the original train dataset
            y_train: Corresponding target labels of x_train

        Prints gradient difference of values calculated via numerical approximation and backprop implementation
    """
    epsilon = 0.01
    layer_index = 0
    in_unit_index = 0
    out_unit_index = 0
    output_string = None

    for i in range(6):  # 2 bias, 2 input weights 2 hidden weights
        model_copy_1 = copy.deepcopy(model)
        model_copy_2 = copy.deepcopy(model)

        if i < 3:   # inner layer weights
            layer_index = 0         # 0: inner layer 1: outer layer
            in_unit_index = 0  # make input or hidden based on layer index
            if i == 0:
                output_string = "Hidden Bias Weight:"
            else:
                in_unit_index = np.random.randint(1, model.layer_specs[0] + 1)  # no of input units
                output_string = "Input to Hidden Weight:"
            out_unit_index = np.random.randint(0, model.layer_specs[1])   # hidden unit
        elif i < 6:
            layer_index = 1
            in_unit_index = 0
            if i == 3:
                output_string = "Output Bias Weight:"
            else:
                in_unit_index = np.random.randint(1, model.layer_specs[1] + 1)  # no of hidden units
                output_string = "Hidden to Output Weight:"
            out_unit_index = np.random.randint(0, model.layer_specs[2])  # output units

        print(layer_index, in_unit_index, out_unit_index)
        # Change the weight (+)
        model_copy_1.layers[layer_index].w[in_unit_index][out_unit_index] = model_copy_1.layers[layer_index].w[in_unit_index][out_unit_index] + epsilon

        # Call forward on model with w + epsilon
        _, plus_epsilon_loss = model_copy_1(x_train, y_train)

        # Change a weight (-)
        model_copy_1.layers[layer_index].w[in_unit_index][out_unit_index] = model_copy_1.layers[layer_index].w[in_unit_index][out_unit_index] - epsilon

        # Call forward on model_copy with w - epsilon
        _, minus_epsilon_loss = model_copy_1(x_train, y_train)

        # Numerical approximation
        num_approx = (plus_epsilon_loss - minus_epsilon_loss) / (2 * epsilon)

        # Forward/backward pass on model with no epsilon
        model_copy_2(x_train, y_train)
        model_copy_2.backward()  # find dw
        gradientToCompare = model_copy_2.layers[layer_index].dw[in_unit_index][out_unit_index]

        # Compare
        print(output_string)
        print(f"Approximation: {abs(num_approx)}")
        print(f"Gradient: {abs(gradientToCompare)}")
        print(f"Difference: {abs(np.abs(num_approx) - abs(gradientToCompare))}")


def checkGradient(x_train, y_train, config):
    subsetSize = 10  # Feel free to change this
    sample_idx = np.random.randint(0, len(x_train), subsetSize)
    x_train_sample, y_train_sample = x_train[sample_idx], y_train[sample_idx]

    model = Neuralnetwork(config)
    check_grad(model, x_train_sample, y_train_sample)
