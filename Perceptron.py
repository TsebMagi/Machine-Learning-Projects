# Doug Whitley
# PSU Machine Learning Spring 17
# Homework 2

# Imports
import numpy as np

# Constants
INPUT_NUMBER = 784


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Perceptron:
    def __init__(self):
        self.weights = None
        self.output = 0
        self.last_delta = 0

    def setup(self, num_weights):
        self.weights = 2 * np.random.rand(num_weights) - 1

    def receive_input(self, inputs):
        output = sigmoid(np.dot(inputs, self.weights))
        return output

    def update_weights(self, learning_rate, momentum, error):
        self.last_delta = ((learning_rate * self.output * error) + (momentum * self.last_delta))
        self.weights = self.weights + self.last_delta


class PerceptronCluster:

    def __init__(self):
        self.output_layer = None
        self.hidden_layer = None
        self.learning_rate = 0
        self.momentum = 0

    def setup(self, num_outputs, num_hidden, momentum, learning_rate):
        self.output_layer = np.array(num_outputs, Perceptron)
        for p in self.output_layer:
            p.setup(num_hidden)
        self.hidden_layer = np.array(num_hidden, Perceptron)
        for p in self.hidden_layer:
            p.setup(INPUT_NUMBER)
        self.learning_rate = learning_rate
        self.momentum = momentum

    def run_test_epoch(self, inputs):
        pass

    def run_training(self, inputs):
        pass

    def c_matrix(self):
        pass
