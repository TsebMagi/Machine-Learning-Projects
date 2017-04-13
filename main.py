# Doug Whitley
# PSU Machine Learning Spring 17
# Homework 1

# Imports
import Perceptron
import numpy as np

# Project file inputs
BASE_FILES = {'training': "mnist_train.csv", 'testing': "mnist_test.csv"}
# Project constants
RATES = (0.1, 0.01, 0.001)
# Epoch Numbers
START = 1
STOP = 50

# Setup the base perceptron classes
PERCEPTRONS = tuple([(Perceptron.Perceptron(x))for x in range(10)])


# Reads input from file and returns a numpy Array representation of the file
def data_preprocessor(file_name):
    output = []
    with open(file_name, 'r') as in_file:
        output.extend(setup_vector(x) for x in in_file)
    return np.array(output, float)


# Returns a float representation of a comma separated line of values
def process_line(input_line):
    split_line = input_line.split(',')
    return [float(x) for x in split_line]


def run_test(inputs, target_value):
    # Run through the perceptrons giving them the input and expected outceom from the test
    for x in range(10):
        result = PERCEPTRONS[x].receive_input(inputs)
        # Case: Perceptron should have fired and did --> Do nothing
        if x == target_value and result > 0:
            pass
        # Case: Perceptron shouldn't have fired and didn't --> Do nothing
        elif x != target_value and result <= 0:
            pass
        # Case: Perceptron should have fired and didn't --> Update weights
        elif x == target_value and result <= 0:
            PERCEPTRONS[x].update_weights(1, 0)
        # Case: Perceptron shouldn't have fired and did --> Update Weights
        elif x != target_value and result > 0:
            PERCEPTRONS[x].update_weights(0, 1)


def run_accuracy(inputs, target_index):
    # Setup Tracking Vars
    prediction = -10000
    prediction_max = -10000
    # Loop throught hte Perceptrons and pass it the input test
    for x in range(10):
        result = PERCEPTRONS[x].receive_input(inputs)
        # Track which Perceptron fires with the most magnitude
        if result > prediction_max:
            prediction_max = result
            prediction = x
    # Return the Prediction of the Perceptrons and the expected outcome
    return prediction, target_index


def check_accuracy(testing_data_input):
    # Setup the tracking vars
    correct = 0
    total = 0
    # Run through the tests
    for input_data in testing_data_input:
        # Track number of correct by checking expected return
        test_prediction, test_target = run_accuracy(input_data[1:], input_data[0])
        if test_prediction == test_target:
            correct += 1
        total += 1
    # Return the accuracy
    return correct/total


def c_matrix(testing_data_input):
    # Setup Matrix
    confusion_matrix = np.zeros((10, 10), 'int64')
    # run the accuracy and then increment the appropriate matrix cell based on the result of the test
    for data in testing_data_input:
        # Y is the prediction of the test, X is the actual target
        y, x = run_accuracy(data[1:], data[0])
        confusion_matrix[int(x), int(y)] += 1
    return confusion_matrix


def run_epoch(input_data):
    for test in input_data:
        # Slices the input vector to pass in the inputs followed by the target
        run_test(test[1:], test[0])


def setup_vector(input_line):
    # Get a line
    ret = process_line(input_line)
    # Insert the bias into the vector
    ret.insert(1, 1.0)
    # Normalize the vector
    for value in range(2, len(ret)):
        ret[value] /= 255
    # Shouldn't be changed so return an immutable object
    return tuple(ret)


def display_confusion_matrix(to_show):
    print("Confusion Matrix")
    for row in c_matrix(to_show):
        print(row)
    print()

if __name__ == "__main__":
    # Read in file and setup data
    training_data = data_preprocessor(BASE_FILES['training'])
    testing_data = data_preprocessor(BASE_FILES['testing'])
    # Loop through the different rates and display the results
    for rate in RATES:
        # Display the learning rate for the following epochs
        print("Learning Rate: ", rate, '\n')
        # Reset perceptrons for the learning rate
        for x in range(10):
            PERCEPTRONS[x].setup(rate)
        # display base confusion matrix and Calculate base accuracy for epoch 0
        display_confusion_matrix(testing_data)
        print("Epoch, Testing Accuracy, Training Accuracy")
        print('0,', check_accuracy(testing_data), ',', check_accuracy(training_data))
        # Train for the given number of epochs
        for epoch in range(START, STOP):
            # Shuffle the data
            np.random.shuffle(training_data)
            # Run the epoch
            run_epoch(training_data)
            # Display the accuracy based on the testing data after the epoch has run
            print(epoch, ',', check_accuracy(testing_data), ',', check_accuracy(training_data))
        # Display the confusion matrix after training on the given number of epochs
        display_confusion_matrix(testing_data)
        print('\n')
