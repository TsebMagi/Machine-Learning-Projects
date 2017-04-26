import numpy as np

NUM_RAW_INPUT = 784


def sigmoid(x):
    return 1 / (1 + np.exp(-1.0 * x))


class NeuralNetwork:
    def __init__(self, num_hidden, num_output, momentum, learning_rate):

        # Setup information tracking
        self.confusion_matrix = np.zeros((10, 10), int)
        self.targets = np.zeros((10, 10), float)
        self.targets.fill(0.1)
        for x in range(10):
            self.targets[x, x] = 0.9
        # Number of Nodes at each level
        self.num_hidden = num_hidden
        self.num_output = num_output
        # Number of Weights for each level
        self.num_hidden_weights = NUM_RAW_INPUT + 1
        self.num_output_weights = num_hidden + 1
        # Cluster's hyper parameters
        self.momentum = momentum
        self.learning_rate = learning_rate
        # Container for the Hidden Nodes Output
        self.output_layer_input = np.zeros(num_hidden + 1)
        # Setup the Weights for each Layer
        self.hidden_layer = ((np.random.rand(num_hidden, self.num_hidden_weights) / 10) - 0.05)
        self.output_layer = ((np.random.rand(num_output, self.num_output_weights) / 10) - 0.05)
        # Container for the Outputs
        self.hidden_layer_output = np.zeros(num_hidden)
        self.output_layer_output = np.zeros(num_output)
        # Container for the Errors
        self.hidden_layer_errors = np.zeros(num_hidden)
        self.output_layer_errors = np.zeros(num_output)
        # Container for the last set of weight changes
        self.hidden_layer_last_delta = np.zeros((self.num_hidden, self.num_hidden_weights))
        self.output_layer_last_delta = np.zeros((self.num_output, self.num_output_weights))

    def reset(self):
        self.hidden_layer_last_delta = np.zeros(self.num_hidden, self.num_hidden_weights)
        self.output_layer_last_delta = np.zeros(self.num_output, self.num_output_weights)
        self.confusion_matrix = np.zeros((10, 10), dtype=int)

    def run_epoch(self, data, train, matrix):
        correct = 0
        total = len(data)
        for x in data:
            if self.run_trial(x, train, matrix):
                correct += 1
        return correct / total

    def generate_result(self, data):
        self.hidden_layer_output = sigmoid((self.hidden_layer @ data))
        self.output_layer_input = np.insert(self.hidden_layer_output, 0, [1], axis=0)
        self.output_layer_output = sigmoid(self.output_layer @ self.output_layer_input)

    def train(self, target, data):

        self.output_layer_errors = (self.output_layer_output * (1 - self.output_layer_output) *
                                    (self.targets[int(target)] - self.output_layer_output))

        sum_of_weights_to_out = self.output_layer.transpose() @ self.output_layer_errors
        self.hidden_layer_errors = (self.hidden_layer_output * (1 - self.hidden_layer_output) *
                                    sum_of_weights_to_out[1:])

        self.output_layer_last_delta = (((self.output_layer_errors[np.newaxis, :].transpose() @
                                          self.output_layer_input[np.newaxis, :]) * self.learning_rate) +
                                        (self.output_layer_last_delta * self.momentum))

        self.hidden_layer_last_delta = (((self.hidden_layer_errors[np.newaxis, :].transpose() @ data[np.newaxis, :]) *
                                         self.learning_rate) + (self.hidden_layer_last_delta * self.momentum))

        self.hidden_layer += self.hidden_layer_last_delta
        self.output_layer += self.output_layer_last_delta

    def run_trial(self, trial_data, train, matrix):
        target = trial_data[0]

        self.generate_result(trial_data[1:])
        result = np.argmax(self.output_layer_output)

        if train:
            self.train(target, trial_data[1:])

        if matrix:
            self.confusion_matrix[result, target] += 1

        return target == result
