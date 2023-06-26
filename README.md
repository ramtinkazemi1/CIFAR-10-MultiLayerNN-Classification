# CIFAR-10 Classification with Multi-layer Neural Networks

The goal of this project was to develop and optimize neural network models for accurate classification of objects in the CIFAR-10 dataset. We aimed to train a model that could accurately classify the 60,000 32x32 color images into their respective 10 classes.

## Approach

To achieve our goal, we followed the following approach:

- **Neural Network Architecture:** We implemented a multi-layer neural network with softmax outputs. The network consisted of fully connected layers, and we experimented with different activation functions, including sigmoid and ReLU, to determine their impact on training accuracy.

- **Regularization Techniques:** We explored regularization techniques, such as L2 and L1 regularization, to prevent overfitting and improve generalization performance. By adjusting the regularization factors, we aimed to find the right balance between model complexity and regularization strength.

- **Network Topology:** We investigated the effect of network topology on classification performance. This involved adjusting the number of hidden units and layers in the network to determine their impact on accuracy and training convergence.

- **Training Process:** We utilized mini-batch stochastic gradient descent with momentum for efficient parameter updates during training. We incorporated early stopping using a validation set to prevent overfitting and determined the best model based on the validation performance.


# Check the Report.pdf
