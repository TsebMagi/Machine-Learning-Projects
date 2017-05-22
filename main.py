import pandas as pd
import numpy as np

if __name__ == "__main__":
    # load data
    data = pd.read_csv("optdigits-train.txt", sep=',', index_col=-1, dtype=np.float64, header=None)

    print(data)
