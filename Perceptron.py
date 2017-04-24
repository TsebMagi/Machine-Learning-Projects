# Doug Whitley
# PSU Machine Learning Spring 17
# Homework 2

# Imports
import numpy as np

# Constants
INPUT_NUMBER = 784
NUM_OUTPUT = 10


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Perceptron:
    def __init__(self, num_weights):
        self.weights = ((np.random.rand(num_weights) / 10) - 0.05)
        self.inputs = None
        self.output = 0
        self.last_output = 0
        self.last_delta = 0

    def receive_input(self, inputs):
        self.inputs = np.array(inputs)
        self.last_output = self.output
        self.output = sigmoid(np.dot(self.inputs, self.weights[1:]) + self.weights[0])
        return self.output

    def update_weights(self, learning_rate, momentum, error):
        self.last_delta = ((learning_rate * self.output * error) + (momentum * self.last_delta))
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
        self.output = None
        self.hidden_output = None

    def setup(self, num_hidden, momentum, learning_rate):
        self.output_layer = []
        for _ in range(self.num_output):
            self.output_layer.append(Perceptron(num_hidden + 1))
        self.hidden_layer = []
        for _ in range(num_hidden):
            self.hidden_layer.append(Perceptron(INPUT_NUMBER + 1))
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.num_hidden = num_hidden
        self.confusion_matrix = [[[0] * self.num_output] for _ in range(self.num_output)]

    def update(self, target):
        # calculate output errors
        output_errors = []
        for x in range(self.num_output):
            if float(x) == float(target):
                output_errors.append((self.output[x] * (1 - self.output[x]) * (0.9 - self.output[x])))
            else:
                output_errors.append((self.output[x] * (1 - self.output[x]) * (0.1 - self.output[x])))

        # calculate input to hidden error
        hidden_errors = []
        for x in range(self.num_hidden):
            hidden_calc = (self.hidden_output[x] * (1 - self.hidden_output[x]) * (
                sum([(self.output_layer[y].weights[x + 1] * output_errors[y]) for y in range(self.num_output)])))
            hidden_errors.append(hidden_calc)

        # update output to hidden layer
        for x in range(self.num_output):
            self.output_layer[x].update_weights(self.learning_rate, self.momentum, output_errors[x])
        # update input to hidden  weights
        for x in range(self.num_hidden):
            self.hidden_layer[x].update_weights(self.learning_rate, self.momentum, hidden_errors[x])

    def run_trial(self, data, target, train):
        # Present data to hidden layer
        self.hidden_output = np.array([x.receive_input(data) for x in self.hidden_layer])
        assert len(self.hidden_output) == self.num_hidden
        # Present hidden results to output
        self.output = [x.receive_input(self.hidden_output) for x in self.output_layer]
        assert len(self.output) == self.num_output
        if train:
            self.update(target)
        largest = -999999999
        ret = -1
        for x in range(len(self.output)):
            if self.output[x] > largest:
                ret = x
                largest = self.output[x]
        return ret

    def run_epoch(self, inputs, train, matrix):
        # Calculate Output
        total = 0
        correct = 0
        for x in inputs:
            target = x[0]
            result = self.run_trial(x[1:], x[0], train)
            # print("Target: ", target, "Result: ", result)
            if float(target) == float(result):
                correct += 1
            total += 1
            # Update matrix
            if matrix:
                self.c_matrix(target, result)
        return correct / total

    def c_matrix(self, y, x):
        self.confusion_matrix[y][x] += 1
