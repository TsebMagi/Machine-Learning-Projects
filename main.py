import pandas as pd
import numpy as np


def create_clusters(data, size):
    return np.random.choice(data, size, replace=False)


def run_k_means(data, clusters, ):
    pass


if __name__ == "__main__":
    # load data
    train_data = pd.read_csv("optdigits-train.txt", sep=',', index_col=-1, dtype=np.float64, header=None)
    test_data = pd.read_csv("optdigits-test.txt", sep=',', index_col=-1, dtype=np.float64, header=None)
