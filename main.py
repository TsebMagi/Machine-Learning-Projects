import pandas as pd
import sklearn.model_selection
from math import sqrt, e, pi, log, exp, pow
import numpy

if __name__ == "__main__":
    # Load and process data in the manner that Mike prescribed
    data = pd.read_csv("spambase.data", header=None, index_col=57)
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(data, data.index.values, test_size=0.5)
    # Setup base probabilities
    p_spam = sum(y_train) / len(y_train)
    p_not_spam = 1 - p_spam

    # Calculate standard deviations and means for the training features
    std_spam = x_train.loc[1].std(axis=0).replace(0, 0.0001).as_matrix()
    std_not_spam = x_train.loc[0].std(axis=0).replace(0, 0.0001).as_matrix()
    mean_spam = x_train.loc[1].mean(axis=0).as_matrix()
    mean_not_spam = x_train.loc[0].mean(axis=0).as_matrix()
    # Gauss 1/(sqrt(2*pi)*s)*e**(-0.5*(float(x-m)/s)**2)
    print(x_train)
    print(std_spam)
    print(std_not_spam)
    print(mean_spam)
    print(mean_not_spam)
    print(len(x_test), len(x_train))
    for row in range(len(x_test)):
        spam_class = log(p_spam)
        for x in range(57):
            p_spam += log(1 / (sqrt(2 * pi) * std_spam[x]) * e ** (
            -0.5 * (float(x_test.iloc[row, x] - mean_spam[x]) / std_spam[x]) ** 2))
        pass
        # Calculate parts of the Gauss equation
