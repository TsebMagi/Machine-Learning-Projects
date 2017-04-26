# Doug Whitley
# PSU Machine Learning Spring 17
# Homework 2

# Imports
import Cluster
import numpy as np

# Project file inputs
BASE_FILES = {'training': "mnist_train.csv", 'testing': "mnist_test.csv"}
# Project constants
RATES = (0.1, 0.01, 0.001)
# Epoch Numbers
STOP = 50

if __name__ == "__main__":
    # Read in file and setup data
    training_data = np.genfromtxt(BASE_FILES['training'], delimiter=',')
    testing_data = np.genfromtxt(BASE_FILES['testing'], delimiter=',')

    # Scale Inputs
    for x in training_data:
        x[1:] /= 255
    for x in testing_data:
        x[1:] /= 255

    # Insert Bias
    training_data = np.insert(training_data, 1, [1], axis=1)
    testing_data = np.insert(testing_data, 1, [1], axis=1)

    # Run Experiment 1
    print("Experiment 1: Varying Hidden Units")
    for num in [20, 50, 100]:
        print("=====", num, "=====")
        p_cluster = Cluster.NeuralNetwork(num, 10, 0.9, 0.1)

        for x in range(STOP):
            # Calculate Test and Training on repeat
            testing_accuracy = p_cluster.run_epoch(testing_data, False, False)
            training_accuracy = p_cluster.run_epoch(training_data, True, False)
            print(x, ',', training_accuracy, ',', testing_accuracy)

        p_cluster.run_epoch(testing_data, False, True)
        print(p_cluster.confusion_matrix)

    print("Experiment 2: Varying Momentum")
    for momentum in [0, 0.25, 0.5]:
        print("=====", momentum, "=====")
        p_cluster = Cluster.NeuralNetwork(100, 10, momentum, 0.1)

        for x in range(STOP):
            # Calculate Test and Training on repeat
            testing_accuracy = p_cluster.run_epoch(testing_data, False, False)
            training_accuracy = p_cluster.run_epoch(training_data, True, False)
            print(x, ',', training_accuracy, ',', testing_accuracy)

        p_cluster.run_epoch(testing_data, False, True)
        print(p_cluster.confusion_matrix)

    print("Experiment 3: Partial Data")
    for step in [2, 4]:
        print("=====", step, "=====")
        p_cluster = Cluster.NeuralNetwork(100, 10, 0.9, 0.1)

        for x in range(STOP, step=step):
            # Calculate Test and Training on repeat
            testing_accuracy = p_cluster.run_epoch(testing_data, False, False)
            training_accuracy = p_cluster.run_epoch(training_data, True, False)
            print(x, ',', training_accuracy, ',', testing_accuracy)

        p_cluster.run_epoch(testing_data, False, True)
        print(p_cluster.confusion_matrix)
