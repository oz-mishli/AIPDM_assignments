import time

import gym
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import pickle as pkl
import os

POSITION_GAUSS_CENTERS = 4
VELOCITY_GAUSS_CENTERS = 8
NUM_WEIGHTS_PER_ACTION = POSITION_GAUSS_CENTERS * VELOCITY_GAUSS_CENTERS
COV_MATRIX_INV = np.linalg.inv(np.diag([0.04, 0.0004]))
CENTERS = np.zeros((2, POSITION_GAUSS_CENTERS * VELOCITY_GAUSS_CENTERS))
BEST_W_FILE = 'best_w.pkl'

MAX_STEPS_PER_EPISODE = 200
MAX_TOTAL_STEPS = 4e5
NUM_SIMULATIONS = 100
STEPS_FOR_TEST = 10000
EPSILON_DECAY = 1e-6
MIN_EPSILON = 0.05

ACTION_NAME_MAPPING = {
    0: 'la',
    1: 'na',
    2: 'ra'
}


def generate_centers(p_range, v_range):
    """
    Generate random centers for the RBF features
    :param p_range: tuple of (a, b) representing the range of positions available
    :param v_range: tuple if (a, b) representing the range of speed values
    :return: Numpy array with all the circle centers of shape (2, POSITION_GAUSS_CENTERS * VELOCITY_GAUSS_CENTERS) This
    array contains all possible combinations of position circles and velocity circles (i.e. cartersian product)
    """
    rng = np.random.default_rng()

    c_p = rng.uniform(p_range[0], p_range[1], POSITION_GAUSS_CENTERS)
    c_v = rng.uniform(v_range[0], v_range[1], VELOCITY_GAUSS_CENTERS)

    print(f'Position centers: {c_p}\nVelocity centers: {c_v}')
    return np.transpose(np.array(list(product(c_p, c_v))))


def generate_uniform_centers(p_range, v_range):
    """
    Generate fixed centers for the RBF features, evenly distributed within the respective ranges for position and
    velocity
    :param p_range: tuple of (a, b) representing the range of positions available
    :param v_range: tuple if (a, b) representing the range of speed values
    :return: Numpy array with all the circle centers of shape (2, POSITION_GAUSS_CENTERS * VELOCITY_GAUSS_CENTERS) This
    array contains all possible combinations of position circles and velocity circles (i.e. cartersian product)
    """

    p_step = (p_range[1] - p_range[0]) / POSITION_GAUSS_CENTERS
    v_step = (v_range[1] - v_range[0]) / VELOCITY_GAUSS_CENTERS
    c_p = np.arange(p_range[0], p_range[1], p_step)
    c_v = np.arange(v_range[0], v_range[1], v_step)

    print(f'Position centers: {c_p}\nVelocity centers: {c_v}')
    return np.transpose(np.array(list(product(c_p, c_v))))


def state_to_feature_vector(s):
    """
    Translate from a given state to a feature vector representing that state
    :param s: Input state
    :return: The feature vector represrnting the input state
    """
    x = s[..., np.newaxis] - CENTERS
    return np.diag(np.exp(-0.5 * (np.transpose(x) @ COV_MATRIX_INV @ x)))


def Q(s, a, w):
    """
    Q value function (estimate)
    :param s: State
    :param a: Action
    :param w: Weights of the currert Q function
    :return:
    """

    # Calculate the feature vector
    theta = state_to_feature_vector(s)

    # Multiply by the relevant action weights
    return np.dot(theta, w[a])


def choose_best_action(s, w):
    """
    Efficient function to calculate greedy action
    :param s: current state
    :param w: policy value function parameters vector
    :return: The best action according to current Q function
    """
    # Calculate the feature vector
    theta = state_to_feature_vector(s)

    return np.argmax([np.dot(theta, w[0]),
                      np.dot(theta, w[1]),
                      np.dot(theta, w[2])])


def epsilon_greedy_next_action(env, rng: np.random.Generator, epsilon, Q, current_state, w):
    """
    Choose the next action in an epsilon-greedy manner
    :param env: current RL environment
    :param rng: NumPy random number generator
    :param epsilon: Epsilon
    :param Q: Q function approximation
    :param current_state: The current state
    :param w: Weights
    :return: The action selected according to epsilon-greedy policu
    """
    if rng.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # exploration
    else:
        return choose_best_action(current_state, w)


def control_algorithm(environment, gamma=1, alpha=0.0005, lambda_val=0.5, epsilon=0.995,
                      verbose=False, plot=False):
    """
    Executing the control algorithm for solving the Mountain Car environment using Sarsa Lambda algorithm
    :param environment: the Mountain Car environment
    :param gamma: Gamma value
    :param alpha: Alpha value
    :param lambda_val: Lambda value
    :param epsilon: Epsilon value
    :param verbose:
    :param plot: Whether to plot a chart summarizing the learning process
    :return: The best weights of the Q function as determined by the control algorithm
    """

    # Random init
    rng = np.random.default_rng()

    # Init weights
    n_actions = environment.action_space.n
    w = np.zeros((n_actions, NUM_WEIGHTS_PER_ACTION))

    total_steps = 0
    num_episodes = 0
    tests_counter = 1
    curr_epsilon = epsilon
    simulations_steps = list()
    simulations_mean_reward = list()

    max_avg_reward = -np.Infinity
    best_w = None
    last_E = 0

    while total_steps < MAX_TOTAL_STEPS:

        curr_state = environment.reset()
        total_rewards = 0
        for i in range(MAX_STEPS_PER_EPISODE):
            # Choose next action according to epsilon-greedy
            action = epsilon_greedy_next_action(env, rng, curr_epsilon, Q, curr_state, w)

            # Perform action
            next_state, reward, done, prob = environment.step(action)
            total_steps += 1

            # Calculate delta
            old_q_val = Q(curr_state, action, w)
            # Q learning
            # delta = (reward + gamma * np.max(
            #    [Q(next_state, 0, w), Q(next_state, 1, w), Q(next_state, 2, w)])) - old_q_val

            # Backward-view TD lambda
            next_action = epsilon_greedy_next_action(env, rng, curr_epsilon, Q, next_state, w)
            delta = reward + gamma * Q(next_state, next_action, w) - old_q_val
            curr_E = gamma * lambda_val * last_E + state_to_feature_vector(curr_state)

            # SGD step
            # w[action] += alpha * delta * state_to_feature_vector(curr_state) # Q-learning
            w[action] += alpha * delta * curr_E  # Backward-view TD lambda

            # Eligibility traces update
            last_E = curr_E

            total_rewards += reward

            # Check if episode is done
            if done:
                break

            curr_state = next_state

        num_episodes += 1

        # Simulate current policy
        if total_steps > STEPS_FOR_TEST * tests_counter:
            tests_counter += 1
            mean_reward = simulate_policy(environment, w, verbose=verbose, render=False)
            simulations_steps.append(STEPS_FOR_TEST * tests_counter)
            simulations_mean_reward.append(mean_reward)

            # Store the policy which scored best mean reward (as the convergence process is rather noisy)
            if mean_reward > max_avg_reward:
                max_avg_reward = mean_reward
                best_w = w
            print(f'mean reward = {mean_reward}, epsilon = {curr_epsilon}')

        # Update exploration with exponential decay
        # curr_epsilon = epsilon * np.exp(-EPSILON_DECAY * num_episodes)
        curr_epsilon = np.max([epsilon - EPSILON_DECAY * total_steps, MIN_EPSILON])

    if plot:
        plt.plot(simulations_steps, simulations_mean_reward)
        plt.xlabel('Control Steps')
        plt.ylabel('Average Mean Reward')
        plt.show()

    print(f'Sarsa Lambda finished, total steps = {total_steps}, mean reward for best policy = {max_avg_reward}')
    return best_w


def simulate_policy(env, w, num_trials=NUM_SIMULATIONS, verbose=False, render=False, for_latex=False):
    total_rewards = 0

    for i in range(num_trials):
        ep_reward = 0
        steps = 0
        curr_state = env.reset()

        while True:
            if render:
                env.render()
                time.sleep(0.025) # Add delay so we can meaningfully watch the episode
            action = choose_best_action(curr_state, w)

            # Action
            next_state, reward, done, prob = env.step(action)
            steps += 1
            ep_reward += reward

            if verbose:
                if for_latex:
                    print(f'\\item{{}} {next_state[0]}, {next_state[1]}, 0.5, 0, {ACTION_NAME_MAPPING[action]}, {reward}')
                else:
                    print(f'{steps}. {next_state[0]}, {next_state[1]}, 0.5, 0, {ACTION_NAME_MAPPING[action]}, {reward}')

            curr_state = next_state

            if done:
                total_rewards += ep_reward
                if verbose:
                    print(f'Trial episode {i} finished with reward {ep_reward}')
                break

    mean_reward = total_rewards / num_trials
    return mean_reward


def save_w(w, path_to_save):
    with open(path_to_save, 'wb') as to_save_file:
        pkl.dump(w, to_save_file)
    print(f'The best policy was saved to {path_to_save}')


def load_w(path_to_load):
    if os.path.isfile(path_to_load):
        with open(path_to_load, 'rb') as w_file:
            return pkl.load(w_file)
            print(f'The best policy was loaded from {path_to_load}')
    else:
        return None




if __name__ == '__main__':

    env = gym.make('MountainCar-v0')
    best_weights = load_w(BEST_W_FILE)
    CENTERS = generate_uniform_centers((env.env.min_position, env.env.max_position),
                                       (-env.env.max_speed, env.env.max_speed))
    if best_weights is None:

        best_weights = control_algorithm(env, plot=True)
        save_w(best_weights, BEST_W_FILE)

    simulate_policy(env, best_weights, num_trials=10, render=True, verbose=True, for_latex=False)



