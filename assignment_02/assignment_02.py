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
    0: 'move_left',
    1: 'move_down',
    2: 'move_right',
    3: 'move_up'
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


def q_learning(env, gamma=0.95, alpha=0.03, epsilon=0.995, lambda_value=0.1, with_eligibilty_traces=False):
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
            mean_reward = simulate_policy(env, Q, verbose=True, render=False)
            init_state_values.append([STEPS_FOR_TEST * tests_counter, mean_reward])
            print(f'mean reward = {mean_reward}, epsilon = {curr_epsilon}')

        # Update exploration with exponential decay
        # curr_epsilon = epsilon * np.exp(-EPSILON_DECAY * num_episodes)
        curr_epsilon = np.max([epsilon - EPSILON_DECAY * total_steps, MIN_EPSILON])

    # print(Q)
    print(f'Total steps = {total_steps}')

    return np.array(init_state_values)


def simulate_policy(env, Q, num_trials=NUM_SIMULATIONS, verbose=False, render=False):
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
            steps_data.append(
                f'{steps}. {row},{col},{tile_desc}, {env.env.nrow - 1},{env.env.ncol - 1} {ACTION_NAME_MAPPING[action]}, {reward}')
            ep_reward += reward

            curr_state = next_state
            if render:
                env.render()

            if done:
                if verbose:
                    # Documenting the final state
                    (row, col), tile_desc = decode_env_state(env)
                    steps_data.append(
                        f'--> {row},{col},{tile_desc}, {env.env.nrow - 1},{env.env.ncol - 1} DONE, N/A')
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


def plot_policy_iteration(init_state_values):
    """
    Plot the policy iteration convergence diagram
    :param init_state_values: A list with the mean reward of X simulations per step
    """

    plt.plot(init_state_values[:, 0], init_state_values[:, 1])
    plt.xlabel('Improvement iterations')
    plt.ylabel(f'Policy value over {NUM_SIMULATIONS} simulations')

    plt.show()


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


if __name__ == '__main__':
    env = gym.make('FrozenLake8x8-v1')
    state = env.reset()

    init_state_vals = q_learning(env, with_eligibilty_traces=True)

    plot_policy_iteration(init_state_vals)
