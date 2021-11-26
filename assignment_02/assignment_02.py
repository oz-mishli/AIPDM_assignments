import itertools

import gym
import numpy as np
from matplotlib import pyplot as plt
from typing import List

# Constants
NUM_STATES = 8 * 8
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAX_STEPS_PER_EPISODE = 250
MAX_TOTAL_STEPS = 1e6
MAX_EPISODES = 10000
NUM_SIMULATIONS = 8 * 8 * 4
STEPS_FOR_TEST = 10000
EPSILON_DECAY = 1e-6
MIN_EPSILON = 0.05

ACTION_NAME_MAPPING = {
    0: 'left',
    1: 'down',
    2: 'right',
    3: 'up'
}


# Good values for Q-learning
# alpha=0.03
# epsilon=0.995
# EPSILON_DECAY = 1e-6
# MIN_EPSILON = 0.05

# Good values for Q-learning with eligibility traces
# alpha = 0.03
# epsilon=0.995
# lambda_value=0.1
# EPSILON_DECAY = 1e-6
# MIN_EPSILON = 0.05


def q_learning(env, gamma=0.95, alpha=0.03, epsilon=0.995, lambda_value=0.1, with_eligibilty_traces=False,
               verbose=False, plot=False):
    """
    Execute the Q learning algorithm on the Frozen Lake environment
    :param env: an object of the Frozen Lake environment
    :param gamma: Gamma value
    :param alpha: Alpha value
    :param epsilon: Initial epsilon value
    :param lambda_value: Lambda value (relevant only with eligibility traces)
    :param with_eligibilty_traces: whether to use eligibility traces or not
    :param verbose: verbose output during simulation runs
    :param plot: whether to plot a chart after learning ends
    :return: best Q(s, a) array, chart plot (None if plot=False)
    """
    # Random init
    rng = np.random.default_rng()

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Init Q to small random values uniformly distributed
    Q = rng.uniform(0, 0.001, (n_states, n_actions))

    # Init E to zeros
    E = np.zeros((n_states, n_actions))
    total_steps = 0
    num_episodes = 0
    tests_counter = 1
    curr_epsilon = epsilon
    init_state_values = []
    max_avg_reward = 0
    best_Q = None

    while total_steps < MAX_TOTAL_STEPS:

        curr_state = env.reset()
        total_rewards = 0
        for i in range(MAX_STEPS_PER_EPISODE):

            # Choose next step according to epsilon-greedy
            if rng.uniform(0, 1) < curr_epsilon:
                action = env.action_space.sample()  # exploration
            else:
                action = np.argmax(Q[curr_state, :])  # greedy

            next_state, reward, done, prob = env.step(action)
            total_steps += 1

            old_q_val = Q[curr_state, action]
            delta = (reward + gamma * np.max(Q[next_state, :]) - old_q_val)

            if with_eligibilty_traces:
                E[curr_state, action] += 1
                Q[curr_state, action] = old_q_val + alpha * delta * E[curr_state, action]
                E *= lambda_value * gamma
            else:
                Q[curr_state, action] = old_q_val + alpha * delta

            total_rewards += reward

            # Check if episode is done
            if done:
                break

            curr_state = next_state

        num_episodes += 1

        # Simulate current policy
        if total_steps > STEPS_FOR_TEST * tests_counter:
            tests_counter += 1
            mean_reward = simulate_policy(env, Q, verbose=verbose, render=False)
            init_state_values.append([STEPS_FOR_TEST * tests_counter, mean_reward])

            # Store the policy which scored best mean reward (as the convergence process is rather noisy)
            if mean_reward > max_avg_reward:
                max_avg_reward = mean_reward
                best_Q = Q
            print(f'mean reward = {mean_reward}, epsilon = {curr_epsilon}')

        # Update exploration with exponential decay
        # curr_epsilon = epsilon * np.exp(-EPSILON_DECAY * num_episodes)
        curr_epsilon = np.max([epsilon - EPSILON_DECAY * total_steps, MIN_EPSILON])

    print(f'Q-learning finished, total steps = {total_steps}, mean reaward for best policy = {max_avg_reward}')

    if plot:
        plt = plot_q_learning(np.array(init_state_values), alpha, lambda_value, with_eligibilty_traces)
    else:
        plt = None

    return best_Q, plt


def simulate_policy(env, Q, num_trials=NUM_SIMULATIONS, gamma=0.95, verbose=False, render=False, for_latex=False):
    """
    Simulate a policy on the frozen lake challenge
    :param env: an object of the Frozen Lake environment
    :param Q: The Q array of the policy to use
    :param num_trials: number of simulations
    :param gamma: Discount factor parameter.
    :param verbose: verbose output
    :param render: rendering the game step by step
    :param for_latex: output for latex file
    :return: mean total reward of this policy
    """
    total_rewards = 0

    for i in range(num_trials):
        steps_data = list()
        ep_reward = 0
        steps = 0
        curr_state = env.reset()

        while True:

            action = np.argmax(Q[curr_state, :])  # greedy

            # Get current env state data before performing the action
            (row, col), tile_desc = decode_env_state(env)

            # Action
            next_state, reward, done, prob = env.step(action)
            steps += 1

            # Documenting step
            if not for_latex:
                steps_data.append(
                    f'{steps}. [{row},{col}],{tile_desc}, [{env.env.nrow - 1},{env.env.ncol - 1}] {ACTION_NAME_MAPPING[action]}, {reward}')
            else:
                steps_data.append(
                    f'\\item{{}} [{row},{col}],{tile_desc}, [{env.env.nrow - 1},{env.env.ncol - 1}] {ACTION_NAME_MAPPING[action]}, {reward}')
            ep_reward += reward * (gamma ** steps)

            curr_state = next_state
            if render:
                env.render()

            if done:
                if verbose:
                    # Documenting the final state
                    (row, col), tile_desc = decode_env_state(env)
                    if not for_latex:
                        steps_data.append(
                            f'--> [{row},{col}],{tile_desc}, [{env.env.nrow - 1},{env.env.ncol - 1}] DONE, N/A')
                    else:
                        steps_data.append(
                            f'\\item{{}} --> [{row},{col}],{tile_desc}, [{env.env.nrow - 1},{env.env.ncol - 1}] DONE, N/A')
                    format_simulation_print(ep_reward, steps, steps_data)
                    print(f'Finished episode {i} with reward {ep_reward} after {steps} steps')

                total_rewards += ep_reward
                break

    mean_reward = total_rewards / num_trials
    return mean_reward


def decode_env_state(env):
    """
    Decode current state of the Frozen Lake environment, i.e. player location in 0-based indexing and the type of tile
    in this location
    :param env: the Frozen Lake environment object
    :return: (current row, current column), type of tile
    """
    row, col = env.env.s // env.env.ncol, env.env.s % env.env.ncol
    tile_desc = env.env.desc[row][col].decode()
    return (row, col), tile_desc


def plot_q_learning(init_state_values, alpha_val, lambda_val, with_eligibility):
    """
    Plot the policy iteration convergence diagram
    :param init_state_values: A list with the mean reward of X simulations per step
    """

    plt.plot(init_state_values[:, 0], init_state_values[:, 1])
    plt.xlabel('Improvement iterations')
    plt.ylabel(f'Policy value over {NUM_SIMULATIONS} simulations')
    if with_eligibility:
        plt.title(f'Q learning w/ Eligibility | alpha = {alpha_val}, lambda = {lambda_val}')
    else:
        plt.title(f'Q learning w/o Eligibility | alpha = {alpha_val}, lambda = {lambda_val}')

    return plt


def format_simulation_print(total_reward: float, total_steps: int, steps: List):
    """
    Format printing of the simulation outcome.
    :param total_reward: Episode accumulated rewards.
    :param total_steps: Episode total steps executed.
    :param steps: Steps information through the episode execution.
    """
    print(f'total steps: {total_steps}')
    print(f'total rewards: {total_reward}')
    print('\n'.join(steps))


def plot_q_learning_cross_validation(env, hyperparams: [{'alpha': 0, 'lambdas': 0}]):
    num_combs = len(hyperparams)
    f, axarr = plt.subplots(num_combs, 1, squeeze=False)

    # Iterate through all hyperparameters combinations listed
    for i in range(num_combs):
        # Run Q-learning with eligibility traces and plot the chart
        _, axarr[i] = q_learning(env, alpha=hyperparams[i]['alpha'], lambda_value=hyperparams[i]['lambda'],
                                 with_eligibilty_traces=True, verbose=False, plot=True)

    plt.show()


if __name__ == '__main__':
    env = gym.make('FrozenLake8x8-v1')

    # Learn with 4 parameter combinations
    plot_q_learning_cross_validation(env, hyperparams=[{'alpha': 0.1, 'lambda': 0.15},
                                                       {'alpha': 0.03, 'lambda': 0.08},
                                                       {'alpha': 0.15, 'lambda': 0.5},
                                                       {'alpha': 0.5, 'lambda': 0.15}])

    chosen_alpha = 0.03
    chosen_lambda = 0.08

    # Use best combination of parameters to create the best policy
    Q, _ = q_learning(env, alpha=chosen_alpha, lambda_value=chosen_lambda, with_eligibilty_traces=False, plot=True)

    simulate_policy(env, Q, num_trials=3, verbose=True, for_latex=True)
