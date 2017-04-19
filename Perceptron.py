# Doug Whitley
# PSU Machine Learning Spring 17
# Homework 1

# Imports
import random
import numpy as np
import scipy.special.expit

# Constants
INPUT_NUMBER = 784


class HiddenLayerNode:
    def __init__(self):
        self.weights = None
        self.output = 0
        self.last_delta = 0

    def setup(self, momentum):
        self.weights = 2 * np.random.rand(INPUT_NUMBER) - 1

    def receive_input(self, inputs):
        output = np.dot(inputs, self.weights)
        return output

    def update_weights(self, learning_rate, momentum, error):
        self.last_delta = ((learning_rate * self.output * error) + (momentum * self.last_delta))
        self.weights = self.weights + self.last_delta


# Perceptron Class
class Perceptron:
    def __init__(self):
        self.weights = None
        self.target = 0
        self.momentum = 0

    def setup(self, momentum, num_hidden, target):
        self.weights = 2 * np.random.rand(num_hidden) - 1
        self.target = target
        self.momentum = momentum
