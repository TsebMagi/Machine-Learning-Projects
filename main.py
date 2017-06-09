import numpy as np
from enum import IntEnum

CONST_DISCOUNT = 0.9
CONST_ACTIONS = 200


class Rewards(IntEnum):
    HIT_WALL = -5
    PICK_UP_CAN = 10
    PICK_UP_EMPTY = -1


class Actions(IntEnum):
    PICK_UP = 0
    MOVE_NORTH = 1
    MOVE_EAST = 2
    MOVE_SOUTH = 3
    MOVE_WEST = 4


class Directions(IntEnum):
    HERE = 0
    NORTH = 1
    EAST = 2
    SOUTH = 3
    WEST = 4


class BoardItems(IntEnum):
    WALL = 1
    CAN = 2
    NOTHING = 0


def create_states():
    ret = [[(h, n, e, s, w), np.zeros(5)] for h in range(3) for n in range(3) for s in range(3) for e in range(3) for w
           in range(3)]
    return ret


def create_board():
    # Creates a 12 x 12 board with 1's along the exterior and fills it with a random number of cans
    ret = np.zeros((12, 12), dtype='int')
    ret[0, 0] = ret[-1, -1] = ret[-1, 0] = ret[0, -1] = 1
    ret[0, 1:-1] += 1
    ret[-1, 1:-1] += 1
    ret[1:-1, 0] += 1
    ret[1:-1, -1] += 1
    for x in range(1, 11):
        for y in range(1, 11):
            can = np.random.randint(100)
            if can > 49:
                ret[x, y] = 2
    return ret


def check_sensors(x_pos, y_pos, board):
    here = board[x_pos, y_pos]
    north = board[x_pos - 1, y_pos]
    south = board[x_pos + 1, y_pos]
    east = board[x_pos, y_pos + 1]
    west = board[x_pos - 1, y_pos - 1]
    return tuple((here, north, east, south, west))


def take_action(action, c_x, c_y, surroundings, tax, hidden_treasure, board):
    # setup local vars for ease of use
    reward = 0
    new_x = c_x
    new_y = c_y
    # Action Cases with their possible result calculations
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
    # Check for hidden treasure
    if reward < 0 and hidden_treasure is True:
        find = np.random.rand()
        if find > 0.995:
            reward *= -100
    # calculate resulting reward and position
    return reward - tax, new_x, new_y


def robby_loop(states, num_steps, greedy_term, learning_rate, tax, hidden_treasure, training):
    # Setup random starting point for episode
    x = np.random.randint(1, 11)
    y = np.random.randint(1, 11)
    # set rewards to 0
    rewards = 0
    # create random board for episode
    board = create_board()
    # setup probability of random action
    good_action = 1 - greedy_term
    # run episode
    for _ in range(num_steps):
        # check for state
        scanner_input = check_sensors(x, y, board)
        # find current state in stat list
        current_state_pair = None
        for s in states:
            if s[0] == scanner_input:
                current_state_pair = s
        if current_state_pair is None:
            print("Error Current State Not Found")
        # Calculate random or good action
        greedy = np.random.rand()
        if greedy > good_action:
            # Take random action
            action = np.random.randint(5)
        else:
            # Take Good Action
            greatest = -99999
            best_list = None
            list_len = 0
            for thing in enumerate(current_state_pair[1]):
                if thing[1] > greatest:
                    best_list = [thing[0]]
                    greatest = thing[1]
                    list_len = 1
                if thing[1] == greatest:
                    best_list.append(thing[0])
                    list_len += 1
            action = best_list[np.random.randint(list_len)]
        # Calculate result of action
        action_reward, x, y = take_action(action, x, y, scanner_input, tax, hidden_treasure, board)
        rewards += action_reward
        # find new state
        new_state = check_sensors(x, y, board)
        new_state_pair = None
        for s in states:
            if s[0] == new_state:
                new_state_pair = s
        if new_state_pair is None:
            print("Error New State Not Found")
        if training:
            # update Q for state
            (current_state_pair[1])[action] += \
                (learning_rate * (action_reward + CONST_DISCOUNT * (np.amax(new_state_pair[1]))
                                  - current_state_pair[1][action]))
    return rewards


def train_and_test(learning_rates, epsilon_value, tax, treasure, const_ep):
    # Iterate across the learning rates passed in
    for rate in learning_rates:
        # Setup a clean set of states
        states = create_states()
        # Reset Rewards
        train_rewards = 0
        # Reset to the initial Epsilon, in case we need to run multiple rates
        epsilon = epsilon_value
        print("----------training----------")
        print("Rate: ", rate)
        # Iterate across the epochs
        for epoch in range(1, 5001):
            # On the appropriate epochs reduce the epsilon term
            if epoch % 50 == 0 and epsilon > 0.1 and const_ep is False:
                epsilon -= 0.01
            # Calculate the rewards for the episode
            train_rewards += robby_loop(states, CONST_ACTIONS, epsilon, rate, tax, treasure, True)
            # Produce plot point at appropriate intervals
            if epoch % 100 == 0:
                print(epoch, ",", train_rewards)
        # Run the Testing side
        print("----------testing----------")
        print("Rate: ", rate)
        # Array of episode results
        values = np.zeros(5000)
        # Fill array of episode results
        test_rewards = 0
        for epoch in range(5000):
            values[epoch] = robby_loop(states, CONST_ACTIONS, 0, rate, tax, treasure, False)
            test_rewards += values[epoch]
        # Calculate relevant stats based on results
        print("Average: ", np.average(values))
        print("Standard Deviation: ", np.std(values))


if __name__ == "__main__":
    # Run the experiments
    print("-----Experiment 1-----")
    train_and_test(learning_rates=[0.2], epsilon_value=1.0, tax=0, treasure=False, const_ep=False)

    print("-----Experiment 2-----")
    train_and_test(learning_rates=[0.0001, 0.333, 0.666, 1.0], epsilon_value=1.0, tax=0, treasure=False, const_ep=False)

    print("-----Experiment 3-----")
    train_and_test(learning_rates=[0.2], epsilon_value=0.33, tax=0, treasure=False, const_ep=True)

    print("-----Experiment 4-----")
    train_and_test(learning_rates=[0.2], epsilon_value=1.0, tax=0.5, treasure=False, const_ep=False)

    print("-----Experiment 5-----")
    train_and_test(learning_rates=[0.2], epsilon_value=1.0, tax=0, treasure=True, const_ep=False)
