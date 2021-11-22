import gym
import numpy as np
from matplotlib import pyplot as plt

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
ALPHA_DECAY = 1e-8
MIN_EPSILON = 0.001


# alpha = 0.03
def q_learning(env, gamma=0.95, alpha=0.03, epsilon=0.995, lambda_value=0.9, with_eligibilty_traces=False):
    # Random init
    rng = np.random.default_rng()

    # Initialize Q(s, a) to zeros
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    # Q = np.zeros((n_states, n_actions), dtype=float)
    Q = rng.uniform(0, 0.001, (n_states, n_actions))
    E = np.zeros((n_states, n_actions))
    total_steps = 0
    num_episodes = 0
    tests_counter = 1
    curr_epsilon = epsilon
    curr_alpha = alpha
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
                Q[curr_state, action] = old_q_val + curr_alpha * delta * E[curr_state, action]
                E *= lambda_value * gamma
            else:
                Q[curr_state, action] = old_q_val + curr_alpha * delta

            total_rewards += reward

            # Check if episode is done
            if done:
                break

            curr_state = next_state

        num_episodes += 1

        # Simulate current policy
        if total_steps > STEPS_FOR_TEST * tests_counter:
            tests_counter += 1
            mean_reward = simulate_policy(env, Q)
            init_state_values.append([STEPS_FOR_TEST * tests_counter, mean_reward])
            print(f'mean reward = {mean_reward}, epsilon = {curr_epsilon} alpha={curr_alpha}')

        # Update exploration with exponential decay TBD (epsilon GLIE)
        # curr_epsilon = epsilon * np.exp(-EPSILON_DECAY * num_episodes)
        curr_epsilon = np.max([epsilon - EPSILON_DECAY * total_steps * 1.5, MIN_EPSILON])
        curr_alpha = alpha - ALPHA_DECAY * total_steps

    print(Q)
    print(f'Total steps = {total_steps}')

    return np.array(init_state_values)


def simulate_policy(env, Q, num_trials=NUM_SIMULATIONS, verbose=False, render=False):
    total_rewards = 0
    for i in range(num_trials):

        ep_reward = 0
        steps = 0
        curr_state = env.reset()

        while True:

            action = np.argmax(Q[curr_state, :])  # greedy

            next_state, reward, done, prob = env.step(action)
            steps += 1
            ep_reward += reward
            curr_state = next_state

            if done:
                if verbose:
                    print(f'Finished episode {i} with reward {ep_reward} after {steps} steps')

                total_rewards += ep_reward
                break

    mean_reward = total_rewards / num_trials
    return mean_reward


def plot_policy_iteration(init_state_values):
    """
    Plot the policy iteration convergence diagram
    :param init_state_values: A list with the mean reward of X simulations per step
    """

    plt.plot(init_state_values[:, 0], init_state_values[:, 1])
    plt.xlabel('Improvement iterations')
    plt.ylabel(f'Policy value over {NUM_SIMULATIONS} simulations')

    plt.show()


if __name__ == '__main__':
    env = gym.make('FrozenLake8x8-v1')
    state = env.reset()
    a = env.step(RIGHT)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # print(state)
    # print(env.action_space.n)
    # print(a)
    # print(env.render())

    init_state_vals = q_learning(env, with_eligibilty_traces=True)

    plot_policy_iteration(init_state_vals)
