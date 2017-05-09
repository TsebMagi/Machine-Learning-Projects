import pandas as pd
import random
import sklearn.model_selection
import sklearn.svm
import sklearn.preprocessing


def run_experiment1():
    print("Experiment 1", '\n')
    # Get the relevant stats
    predictions = classifier.predict(x_test)
    probabilities = classifier.predict_proba(x_test)
    acc = classifier.score(x_test, y_test)
    precision = sklearn.metrics.precision_score(y_test, predictions)
    recall = sklearn.metrics.recall_score(y_test, predictions)
    print("Accuracy", acc)
    print("Precision", precision)
    print("Recall", recall)
    # Produce the data for the ROC curve
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, probabilities[:, 1], drop_intermediate=True)
    for x in range(len(thresholds)):
        print(fpr[x], ',', tpr[x])


def run_experiment2():
    print("Experiment 2", '\n')
    # Setup an array of the indices of the features in increasing order of weight
    temp = classifier.coef_
    arr = temp[0].argsort()[-57:][::-1]
    # Run through the features
    for x in range(2, 58):
        # Fit and calculate accuracy for the set of Features
        classifier2 = sklearn.svm.SVC(kernel='linear')
        classifier2.fit(x_train.iloc[:, arr[:x]], y_train)
        sorted_acc = classifier2.score(x_test.iloc[:, arr[:x]], y_test)
        print(x, ',', sorted_acc)


def run_experiment3():
    print("Experiment 3", '\n')
    # Setup an array of random indices
    arr = [x for x in range(57)]
    random.shuffle(arr)
    # Run through the features
    for x in range(2, 58):
        # Fit and calculate accuracy for the set of features
        classifier3 = sklearn.svm.SVC(kernel='linear')
        classifier3.fit(x_train.iloc[:, arr[:x]], y_train)
        rand_acc = classifier3.score(x_test.iloc[:, arr[:x]], y_test)
        print(x, ',', rand_acc)


if __name__ == "__main__":
    # Load and process data in the manner that Mike prescribed
    data = pd.read_csv("spambase.data", header=None, index_col=57)
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(data, data.index.values, test_size=0.5)
    scaler = sklearn.preprocessing.StandardScaler().fit(x_train)

    # Transform the training data based on that scaler and set the index values accordingly
    x_train = pd.DataFrame(scaler.transform(x_train), index=y_train)

    # Transform the testing data based on that same scaler and set the index values accordingly
    x_test = pd.DataFrame(scaler.transform(x_test), index=y_test)
    classifier = sklearn.svm.SVC(kernel='linear', probability=True)
    classifier.fit(x_train, y_train)
    # Run the experiments
    run_experiment1()
    run_experiment2()
    run_experiment3()
