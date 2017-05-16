import pandas as pd
import sklearn.model_selection
from math import sqrt, e, pi, log
import numpy

if __name__ == "__main__":
    # Load and process data in the manner that Mike prescribed
    data = pd.read_csv("spambase.data", header=None, index_col=57)
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(data, data.index.values, test_size=0.5)
    # Setup base probabilities
    p_spam = sum(y_train) / len(y_train)
    p_not_spam = 1 - p_spam

    # Calculate standard deviations and means for the training features
    std_spam = x_train.loc[1].std(axis=0).replace(0, 0.0001)
    std_not_spam = x_train.loc[0].std(axis=0).replace(0, 0.0001)
    mean_spam = x_train.loc[1].mean(axis=0)
    mean_not_spam = x_train.loc[0].mean(axis=0)
    # Setup the tracker variables
    correct = 0
    confusion_matrix = pd.DataFrame(numpy.zeros((2, 2)))
    # Go through the test set
    for row in range(len(x_test)):
        # Setup the base probabilities
        spam_class = log(p_spam)
        not_spam_class = log(p_not_spam)
        # Calculate each Feature
        for x in range(57):
            # Split the log term into two via the log addition property
            spam_class += log(1 / (sqrt(2 * pi) * std_spam[x]))
            # Calculate second term separately to avoid taking log of 0
            second_term_spam = e ** (-0.5 * (float(x_test.iloc[row, x] - mean_spam[x]) / std_spam[x]) ** 2)
            # if second term has gone to 0 use very negative number instead
            if second_term_spam == 0:
                spam_class += -100000000
            else:
                # Didn't go to zero
                spam_class += log(second_term_spam)
            # Repeat above for the not spam case
            not_spam_class += log(1 / (sqrt(2 * pi) * std_not_spam[x]))
            second_term_not_spam = e ** (-0.5 * (float(x_test.iloc[row, x] - mean_not_spam[x]) / std_not_spam[x]) ** 2)
            if second_term_not_spam == 0:
                not_spam_class += -100000000
            else:
                not_spam_class += log(second_term_not_spam)
        # Calculate guess and update stats
        if spam_class > not_spam_class:
            guess = 1
        else:
            guess = 0
        if guess == y_test[row]:
            correct += 1
        confusion_matrix.iloc[guess, y_test[row]] += 1
    # Display results
    print("Accuracy", correct / len(x_train))
    print(confusion_matrix)
