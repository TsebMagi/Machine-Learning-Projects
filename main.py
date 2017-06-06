import numpy as np
import logging
from enum import IntEnum


class Rewards(IntEnum):
    HIT_WALL = -5
    PICK_UP_CAN = 10
    PICK_UP_EMPTY = -1


class Actions(IntEnum):
    PICK_UP = 0
    MOVE_NORTH = 1
    MOVE_SOUTH = 2
    MOVE_EAST = 3
    MOVE_WEST = 4


class Directions(IntEnum):
    HERE = 0
    NORTH = 1
    SOUTH = 2
    EAST = 3
    WEST = 4


class BoardItems(IntEnum):
    WALL = 1
    CAN = 2
    NOTHING = 0


def create_states():
    ret = [[(h, n, s, e, w), np.zeros(5)] for h in range(3) for n in range(3) for s in range(3) for e in range(3) for w
           in range(3)]
    return ret


def create_board():
    ret = np.zeros((12, 12), dtype='int')
    ret[0, 0] = ret[-1, -1] = ret[-1, 0] = ret[0, -1] = 1
    ret[0, 1:-1] += 1
    ret[-1, 1:-1] += 1
    ret[1:-1, 0] += 1
    ret[1:-1, -1] += 1
    for x in range(1, 10):
        for y in range(1, 10):
            can = np.random.randint(100)
            if can > 49:
                ret[x, y] = 2
    return ret


def check_sensors(x_pos, y_pos):
    here = board[x_pos, y_pos]
    north = board[x_pos - 1, y_pos]
    south = board[x_pos + 1, y_pos]
    east = board[x_pos, y_pos + 1]
    west = board[x_pos - 1, y_pos - 1]
    return tuple((here, north, south, east, west))


def take_action(action, c_x, c_y, surroundings):
    reward = 0
    new_x = c_x
    new_y = c_y
    if action == Actions.MOVE_NORTH:
        if surroundings[Directions.NORTH] != BoardItems.WALL:
            new_x -= 1
        else:
            reward = Rewards.HIT_WALL
    elif action == Actions.MOVE_SOUTH:
        if surroundings[Directions.SOUTH] != BoardItems.WALL:
            new_x += 1
        else:
            reward = Rewards.HIT_WALL
    elif action == Actions.MOVE_EAST:
        if surroundings[Directions.EAST] != BoardItems.WALL:
            new_y += 1
        else:
            reward = Rewards.HIT_WALL
    elif action == Actions.MOVE_WEST:
        if surroundings[Directions.WEST] != BoardItems.WALL:
            new_y -= 1
        else:
            reward = Rewards.HIT_WALL
    elif action == Actions.PICK_UP:
        if surroundings[Directions.HERE] == BoardItems.CAN:
            board[new_x, new_y] = 0
            reward = Rewards.PICK_UP_CAN
        else:
            reward = Rewards.PICK_UP_EMPTY
    else:
        print("INVALID ACTION NUMBER: ", action)
    return reward, new_x, new_y


def robby_loop(num_steps, greedy_term, learning_rate, discount):
    x = np.random.randint(1, 10)
    y = np.random.randint(1, 10)
    rewards = 0
    good_action = 1 - greedy_term

    for _ in range(num_steps):
        scanner_input = check_sensors(x, y)
        current_state_pair = None
        for s in states:
            if s[0] == scanner_input:
                current_state_pair = s
        if current_state_pair is None:
            print("Error Current State Not Found")
        greedy = np.random.rand()
        if greedy > good_action:
            # Take random action
            action = np.random.randint(5)
            # print("Random")
        else:
            # Take Good Action
            # print("Good!")
            action = np.argmax(current_state_pair[1])
        action_reward, x, y = take_action(action, x, y, scanner_input)
        rewards += action_reward
        # update Q for state st
        new_state = check_sensors(x, y)
        new_state_pair = None
        for s in states:
            if s[0] == new_state:
                new_state_pair = s
        if new_state_pair is None:
            print("Error New State Not Found")
        (current_state_pair[1])[action] += learning_rate * (action_reward + discount * (np.argmax(new_state_pair[1])
                                                                                        - (current_state_pair[1])[
                                                                                            action]))
    return rewards


if __name__ == "__main__":
    log = logging.getLogger(__name__)
    states = create_states()
    print("Experiment 1")
    epsilon = 1
    learning = 0.2
    disc = 0.9
    train_rewards = 0
    print("----------training----------")
    for epoch in range(1, 5001):
        board = create_board()
        if epoch % 50 == 0 and epsilon > 0.1:
            epsilon -= 0.01
        train_rewards += robby_loop(200, epsilon, learning, disc)
        if epoch % 100 == 0:
            print(epoch, ",", train_rewards)
    test_rewards = 0
    print("----------testing----------")
    values = np.zeros(5000)
    for epoch in range(5000):
        board = create_board()
        values[epoch] = robby_loop(200, 0, learning, disc)
    print("Average: ", np.average(values))
    print("Standard Deviation: ", np.std(values))
