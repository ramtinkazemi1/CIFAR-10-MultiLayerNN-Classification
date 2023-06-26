import numpy as np
import util


class Activation():
    """
    The class implements different types of activation functions for
    your neural network layers.

    """

    def __init__(self, activation_type = "sigmoid"):
        """
        TODO: Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU", "output"]:
            # output can be used for the final layer. Feel free to use/remove it
            raise NotImplementedError(f"{activation_type} is not implemented.")

        # Type of non-linear activation.
        self.activation_type = activation_type
        # Placeholder for input. This can be used for computing gradients.
        self.x = None

        # I think the input will be N examples x dimensions (3072 for first layer maybe 128 for hidden layer)
        # final output should be N x 10 cause 10 classes and

    def __call__(self, z):
        """
        TODO
        This method allows your instances to be callable.
        """
        return self.forward(z)

    def forward(self, z):
        """
        Compute the forward pass.
        """
        if self.activation_type == "sigmoid":
            return self.sigmoid(z)

        elif self.activation_type == "tanh":
            return self.tanh(z)

        elif self.activation_type == "ReLU":
            return self.ReLU(z)

        elif self.activation_type == "output":
            return self.output(z)

    def backward(self, z):
        """
        Compute the backward pass.
        """
        if self.activation_type == "sigmoid":
            return self.grad_sigmoid(z)

        elif self.activation_type == "tanh":
            return self.grad_tanh(z)

        elif self.activation_type == "ReLU":
            return self.grad_ReLU(z)

        elif self.activation_type == "output":
            return self.grad_output(z)

    def sigmoid(self, x):
        """
        TODO: DONE Implement the sigmoid activation here.
        """
        self.x = 1 / (1 + np.exp(-x))
        return self.x

        # raise NotImplementedError("Sigmoid not implemented")

    def tanh(self, x):
        """
        TODO: DONE Implement tanh here.
        """
        self.x = np.tanh(x)
        return self.x

        # raise NotImplementedError("Tanh not implemented")

    def ReLU(self, x):
        """
        TODO: DONE Implement ReLU here.
        """
        self.x = np.maximum(0, x)
        return self.x
        # raise NotImplementedError("ReLU not implemented")

    def output(self, x):
        """
        TODO: DONE Implement softmax function here.
        Remember to take care of the overflow condition.
        """
        x -= np.max(x)
        x_exp = np.exp(x)  # need to either clip it or who knows7
        denominators = np.sum(x_exp, axis=1, keepdims=True)

        self.x = x_exp / denominators
        return self.x

        # raise NotImplementedError("output activation not implemented")

    def grad_sigmoid(self, x):
        """
        TODO: DONE Compute the gradient for sigmoid here.
        """
        # g'(a) = g(a)(1-g(a) for sigmoid
        sig_grad = self.sigmoid(x)*(1 - self.sigmoid(x))
        return sig_grad

        # raise NotImplementedError("Sigmoid gradient not implemented")

    def grad_tanh(self, x):
        """
        TODO: DONE Compute the gradient for tanh here.
        """
        # tanh' = 1 - tanh^2(x)
        tanh_grad = 1 - np.square(np.tanh(x))
        return tanh_grad

        # raise NotImplementedError("Tanh gradient not implemented")

    def grad_ReLU(self, x):
        """
        TODO: DONE Compute the gradient for ReLU here.
        """
        # g'(a) = 1 where g(a) = a and 0 where g(a) = 0
        ReLU_grad = np.where(x == 0, 0, 1)
        return ReLU_grad

        # raise NotImplementedError("ReLU gradient not implemented")

    def grad_output(self, x):
        """
        Deliberately returning 1 for output layer case since we don't multiply by any activation for final layer's delta. Feel free to use/disregard it
        """

        return np.ones(x.shape)


class Layer():
    """
    This class implements Fully Connected layers for your neural network.
    """

    def __init__(self, in_units, out_units, activation, weightType):
        """
        TODO
        Define the architecture and create placeholders.
        """
        np.random.seed(42)
        self.w = None
        if (weightType == 'random'):
            self.w = 0.01 * np.random.random((in_units + 1, out_units))  # first layer 3072 x 128

        self.x = None    # Save the input to forward in this
        self.a = None    # output without activation
        self.z = None    # Output After Activation
        self.activation = activation   # Activation function

        self.dw = 0  # Save the gradient w.r.t w in this. You can have bias in w itself or uncomment the next line and handle it separately
        self.v = 0  # velocity

    def __call__(self, x):
        """
        TODO
        Make layer callable.
        """
        return self.forward(x)

    def forward(self, x):
        """
        TODO: Compute the forward pass (activation of the weighted input) through the layer here and return it.
        """
        appended_x = util.append_bias(x)
        self.x = appended_x
        self.a = self.x @ self.w  # x = N x 3073 w = 3073 x 128 a = N x 128
        # print(self.a.shape)
        self.z = self.activation.forward(self.a)  # z shape = a shape
        # print(self.z.shape)
        return self.z

    def backward(self, deltaCur, learning_rate, momentum_gamma, regularization, gradReqd=True):
        """
        TODO: Write the code for backward pass. This takes in gradient from its next layer as input and
        computes gradient for its weights and the delta to pass to its previous layers. gradReqd is used to specify whether to update the weights i.e. whether self.w should
        be updated after calculating self.dw
        The delta expression (that you prove in PA1 part1) for any layer consists of delta and weights from the next layer and derivative of the activation function
        of weighted inputs i.e. g'(a) of that layer. Hence deltaCur (the input parameter) will have to be multiplied with the derivative of the activation function of the weighted
        input of the current layer to actually get the delta for the current layer. Remember, this is just one way of interpreting it and you are free to interpret it any other way.
        Feel free to change the function signature if you think of an alternative way to implement the delta calculation or the backward pass
        """

        # weight gradient for L2:
        self.dw = (np.dot(self.x.T, (self.activation.backward(self.a) * deltaCur))) / self.x.shape[0] - regularization
        self.v = momentum_gamma * self.v + (1-momentum_gamma) * learning_rate * self.dw

        # weight gradient for L1:
        # self.dw = momentum_gamma*self.dw_prev + learning_rate*(np.dot(self.x.T, delta) - regularization)

        deltaCurNext = np.dot(self.activation.backward(self.a) * deltaCur, self.w[1:, :].T)

        if gradReqd is True:
            self.w = self.w + self.v  # 3072 x N @ N x 128 = 3072 x 128

        return deltaCurNext


class Neuralnetwork():
    """
    Create a Neural Network specified by the network configuration mentioned in the config yaml file.

    """
    def __init__(self, config):
        """
        TODO
        Create the Neural Network using config. Feel free to add variables here as per need basis
        """
        self.layers = []  # Store all layers in this list.
        self.num_layers = len(config['layer_specs']) - 1  # Set num layers here
        self.x = None        # Save the input to forward in this
        self.y = None        # For saving the output vector of the model
        self.targets = None  # For saving the targets
        self.learning_rate = config['learning_rate']
        self.momentum_gamma = config['momentum_gamma']
        self.regularization = config['L2_penalty']  # no idea what this is
        self.batch_size = config['batch_size']
        self.layer_specs = config['layer_specs']

        # Add layers specified by layer_specs.
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                self.layers.append(
                    Layer(config['layer_specs'][i], config['layer_specs'][i + 1], Activation(config['activation']),
                          config["weight_type"]))
            elif i == self.num_layers - 1:
                self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i + 1], Activation("output"),
                                         config["weight_type"]))

    def __call__(self, x, targets=None):
        """
        TODO
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets)
    # make sure targets is passed in as one hot encoded

    def forward(self, x, targets=None):
        """
        TODO: Compute forward pass through all the layers in the network and return the loss.
        If targets are provided, return loss and accuracy/number of correct predictions as well.
        """
        self.x = x
        self.targets = targets
        layer_input = x  # layer input for each layer
        layer_output = None
        # computing forward pass through all the layers
        for i in range(self.num_layers):  # should be 2 if there are len(layer_spec) is 3
            cur_layer = self.layers[i]
            layer_output = cur_layer(layer_input)  # activated output for layer_input

            if i < self.num_layers - 1:
                layer_input = layer_output

        self.y = layer_output  # should be softmax output

        if targets is not None:

            loss = self.loss(self.y, self.targets)  # took mean in loss function
            accuracy = util.calculateCorrect(self.y, self.targets)

            return accuracy, loss

    # logits is just the y
    def loss(self, logits, targets):
        '''
        TODO: compute the categorical cross-entropy loss and return it.
        '''

        return -np.mean(np.sum(targets * np.log(logits + 1e-12), axis=1))

    def backward(self, gradReqd=True):
        '''
        TODO: Implement backpropagation here by calling backward method of Layers class.
        Call backward methods of individual layers.
        '''
        delta = self.targets - self.y

        # for i in range(self.num_layers):
        for i in range(self.num_layers - 1, -1, -1):  # 1, 0
            cur_layer = self.layers[i]
            # calculate delta x weights pass as deltaCur
            # delta_times_weights = delta @ cur_layer.w

            delta_next = cur_layer.backward(delta, self.learning_rate, self.momentum_gamma, self.regularization, True)
            # cur_layer.dw /= self.batch_size
            if i > 0:
                delta = delta_next





