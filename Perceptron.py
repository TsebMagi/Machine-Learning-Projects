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
        output = sigmoid(np.dot(inputs, self.weights[1:])) + self.weights[0]
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

    def run_training(self, data, target):
        # Present data to hidden layer
        hidden_results = np.array([x.receive_input(data) for x in self.hidden_layer])
        # Present hidden results to output
        output = [x.receive_input(hidden_results) for x in self.output_layer]
        # calculate output errors
        output_errors = np.array(
            [x * (1 - x) * (0.9 - x) if output.index(x) == target else x * (1 - x) * (0.2 - x) for x in output])
        # update output to hidden layer
        
        # calculate input to hidden error
        # update input to hidden  weights
        pass

    def run_test(self, data):
        # Present Input to Hidden layer
        # Present result to Output
        pass

    def run_test_epoch(self, inputs, matrix):

        # Calculate Output
        # Update matrix
        pass

    def run_training_epoch(self, inputs):
        for input in inputs:
            self.run_training(input[1:], input[0])

    def c_matrix(self):
        pass
