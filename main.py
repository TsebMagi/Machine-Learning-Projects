# Doug Whitley
# PSU Machine Learning Spring 17
# Homework 2

# Imports
import pandas as pd
import Perceptron
import numpy as np

# Project file inputs
BASE_FILES = {'training': "mnist_train.csv", 'testing': "mnist_test.csv"}
# Project constants
RATES = (0.1, 0.01, 0.001)
# Epoch Numbers
START = 1
STOP = 50

# Reads input from file and returns a numpy Array representation of the file


# Returns a float representation of a comma separated line of values

if __name__ == "__main__":
    # Read in file and setup data
    training_data = np.genfromtxt(BASE_FILES['training'], delimiter=',')
    testing_data = np.genfromtxt(BASE_FILES['testing'], delimiter=',')
    for x in training_data:
        x[1:] /= 255
    for x in testing_data:
        x[1:] /= 255

    # Setup cluster
    p_cluster = Perceptron.PerceptronCluster()
    p_cluster.setup(20, 0.9, 0.1)

    for x in range(STOP):
        # print(p_cluster.run_test_epoch(testing_data,False))
        print(p_cluster.run_epoch(training_data, True, False))
