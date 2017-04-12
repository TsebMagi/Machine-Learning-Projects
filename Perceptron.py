# Doug Whitley
# PSU Machine Learning Spring 17
# Homework 1

# Imports
import random
import numpy as np
# Project Constant
INPUT_NUMBER = 785
# Seed Random
random.seed(a=None)

# Perceptron Class
class Perceptron:

    # Constructor
    def __init__(self,perceptron_class = 0):
        self.target = perceptron_class
        self.weights = None
        self.inputs = None
        self.learningRate = 0

    # Used for testing
    def __str__(self):
        return str(self.target) + str(self.learningRate)

    # Sets up the Perceptron and zeros any previous data it held
    def setup(self, learning_rate):
        self.weights = np.zeros((1,INPUT_NUMBER))
        self.inputs = None
        self.learningRate = learning_rate
        for num in range(INPUT_NUMBER):
            self.weights[0,num] += (random.randrange(-5,5,1)/10)

    # Used for testing
    def display(self):
        print(self.target)
        print(self.inputs)
        print(self.weights)
        print(self.learningRate)
        print(len(self.weights))

    # Receives Input and returns the fire value
    def receive_input(self, input_vector):
        self.inputs = np.array(input_vector)
        return np.dot(self.inputs,self.weights[0])

    # Updates the Weights in the perceptron based on the last set of input
    def update_weights(self, expected_output, actual_output):
        for x in range(len(self.inputs)):
            self.weights[0,x] += self.learningRate * (expected_output - actual_output) * self.inputs[x]
