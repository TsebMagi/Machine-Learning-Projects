import pandas as pd
import numpy as np
from scipy.spatial import distance
import math


def create_board():
    board = np.zeros((12, 12), dtype='int')
    board[0, 0] = board[-1, -1] = board[-1, 0] = board[0, -1] = 1
    board[0, 1:-1] += 1
    board[-1, 1:-1] += 1
    board[1:-1, 0] += 1
    board[1:-1, -1] += 1
    for x in range(1, 10):
        for y in range(1, 10):
            can = np.random.randint(100)
            if can > 49:
                board[x, y] = 2
    return board


if __name__ == "__main__":
    board = create_board()
    print(board)
