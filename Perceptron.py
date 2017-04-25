# Doug Whitley
# PSU Machine Learning Spring 17
# Homework 2

# Imports
from scipy.stats import logistic
import numpy as np
# Constants
INPUT_NUMBER = 784
NUM_OUTPUT = 10


def sigmoid(x):
    return logistic.cdf(x)


class Perceptron:
    def __init__(self, num_weights):
        self.weights = ((np.random.rand(num_weights) / 10) - 0.05)
        self.inputs = None
        self.output = 0
        self.last_output = 0
        self.last_delta = np.zeros(num_weights)

    def receive_input(self, inputs):
        self.inputs = np.array(inputs)
        self.last_output = self.output
        self.output = sigmoid(np.dot(self.inputs, self.weights[1:]) + self.weights[0])
        return self.output

    def reset_momentum(self):
        self.last_delta = np.zeros(len(self.weights))

    def update_weights(self, learning_rate, momentum, error):
        self.last_delta[0] = error * learning_rate + self.last_delta[0] * momentum
        self.last_delta[1:] = (((self.inputs * error) * learning_rate) + (self.last_delta[1:] * momentum))
        self.weights += self.last_delta


class PerceptronCluster:

    def __init__(self):
        self.output_layer = None
        self.hidden_layer = None
        self.learning_rate = 0
        self.momentum = 0
        self.num_hidden = 0
        self.num_output = NUM_OUTPUT
        self.confusion_matrix = None
        self.hidden_output = None
        self.outputs = None

    def setup(self, num_hidden, momentum, learning_rate):
        self.output_layer = [Perceptron(num_hidden + 1) for _ in range(self.num_output)]
        self.hidden_layer = [Perceptron(INPUT_NUMBER + 1) for _ in range(num_hidden)]
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.num_hidden = num_hidden
        self.confusion_matrix = np.zeros((10, 10))

    def update(self, target):
        # calculate output errors
        output_errors = np.array(
            [(self.outputs[x] * (1 - self.outputs[x]) * (0.9 - self.outputs[x])) if x == target else
             (self.outputs[x] * (1 - self.outputs[x]) * (0.1 - self.outputs[x]))
             for x in range(self.num_output)])

        # calculate input to hidden error
        hidden_errors = []
        for x in range(self.num_hidden):
            hidden_errors.append(self.hidden_output[x] * (1 - self.hidden_output[x]) * (
                sum([(self.output_layer[y].weights[x + 1] * output_errors[y]) for y in range(self.num_output)])))
        hidden_errors = np.array(hidden_errors)

        # update output to hidden layer
        for x in range(self.num_output):
            self.output_layer[x].update_weights(self.learning_rate, self.momentum, output_errors[x])
        # update input to hidden  weights
        for x in range(self.num_hidden):
            self.hidden_layer[x].update_weights(self.learning_rate, self.momentum, hidden_errors[x])

    def run_trial(self, data, target, train):
        # Present data to hidden layer
        self.hidden_output = np.array([x.receive_input(data) for x in self.hidden_layer])
        # Present hidden results to output
        self.outputs = np.array([x.receive_input(self.hidden_output) for x in self.output_layer])
        if train:
            self.update(target)
        return np.argmax(self.outputs)

    def run_epoch(self, inputs, train, matrix):
        # Reset epoch tracking vars
        total = 0
        correct = 0

        for p in self.output_layer:
            p.reset_momentum()
        for p in self.hidden_layer:
            p.reset_momentum()

        # Process input data
        for x in inputs:
            # Set target
            target = x[0]
            # Calculate prediction
            result = self.run_trial(x[1:], x[0], train)
            # print("Target: ", target, "Result: ", result)
            # Update tracking vars
            if target == result:
                correct += 1
            total += 1
            # Update matrix
            if matrix:
                self.c_matrix(target, result)
        return correct / total

    def c_matrix(self, y, x):
        self.confusion_matrix[y][x] += 1
