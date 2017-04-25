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

    # Run experiment 1
    print("Experiment 1: Varying Hidden Units")
    for rate in [20, 50, 100]:
        print("=====", rate, "=====")
        p_cluster = Perceptron.PerceptronCluster()
        p_cluster.setup(rate, 0.9, 0.1)

        for x in range(STOP):
            training_accuracy = p_cluster.run_epoch(training_data, True, False)
            testing_accuracy = p_cluster.run_epoch(testing_data, False, False)
            print(x, training_accuracy, testing_accuracy)

        p_cluster.run_epoch(testing_data, False, True)
        print(p_cluster.confusion_matrix)

    print("Experiment 3: Partial Data")
    p_cluster_half = Perceptron.PerceptronCluster()
    p_cluster_half.setup(100, 0.9, 0.1)
    p_cluster_quarter = Perceptron.PerceptronCluster()
    p_cluster_quarter.setup(100, 0.9, 0.1)

    for x in range(STOP):
        training_accuracy = p_cluster.run_epoch(training_data, True, False)
        testing_accuracy = p_cluster.run_epoch(testing_data, False, False)
        print(x, training_accuracy, testing_accuracy)

    p_cluster.run_epoch(testing_data, False, True)
    print(p_cluster.confusion_matrix)
