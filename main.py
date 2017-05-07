import pandas as pd
import random
import sklearn.model_selection
import sklearn.svm
import sklearn.preprocessing


def run_experiment1():
    print("Experiment 1", '\n')
    predictions = classifier.predict(x_test)
    probabilities = classifier.predict_proba(x_test)
    acc = classifier.score(x_test, y_test)
    precision = sklearn.metrics.precision_score(y_test, predictions)
    recall = sklearn.metrics.recall_score(y_test, predictions)
    print("Accuracy", acc)
    print("Precision", precision)
    print("Recall", recall)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, probabilities[:, 1], drop_intermediate=True)
    for x in range(len(thresholds)):
        print(fpr[x], ',', tpr[x])


def run_experiment2():
    print("Experiment 2", '\n')
    arr = classifier.coef_
    arr = arr.argsort()[-57:][::-1]
    for x in range(2, 57):
        classifier2 = sklearn.svm.SVC(kernel='linear')
        classifier2.fit(x_train.iloc[:, arr[0, :x]], y_train)
        sorted_acc = classifier2.score(x_test.iloc[:, arr[0, :x]], y_test)
        print(x, ',', sorted_acc)


def run_experiment3():
    print("Experiment 3", '\n')
    arr = [random.randint(0, 56) for _ in range(57)]
    for x in range(2, 57):
        classifier3 = sklearn.svm.SVC(kernel='linear')
        classifier3.fit(x_train.iloc[:, arr[:x]], y_train)
        sorted_acc = classifier3.score(x_test.iloc[:, arr[:x]], y_test)
        print(x, ',', sorted_acc)


if __name__ == "__main__":
    data = pd.read_csv("spambase.data", header=None, index_col=57)
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(data, data.index.values, test_size=0.5)
    scaler = sklearn.preprocessing.StandardScaler().fit(x_train)

    # Transform the training data based on that scaler and set the index values accordingly
    x_train = pd.DataFrame(scaler.transform(x_train), index=y_train)

    # Transform the testing data based on that same scaler and set the index values accordingly
    x_test = pd.DataFrame(scaler.transform(x_test), index=y_test)
    classifier = sklearn.svm.SVC(kernel='linear', probability=True)
    classifier.fit(x_train, y_train)
    run_experiment1()
    run_experiment2()
    run_experiment3()
