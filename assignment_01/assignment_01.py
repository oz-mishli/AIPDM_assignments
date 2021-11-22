import os.path
import numpy as np
import gym
from matplotlib import pyplot as plt
import pickle as pkl


# Constants
NUM_STATES = 500
NUM_ACTIONS = 6
PROBABILITY = 1.0
GAMMA = 0.95
EPSILON = 1e-4
TEST_TRIALS = 100
BEST_POLICY_FILE = './best_policy.pkl'
STATE_DICT = {0: "move south",
              1: "move north",
              2: "move east",
              3: "move west",
              4: "pickup passenger",
              5: "drop off passenger"}


class MarkovDecisionProcess:
    """
    This class represents the Markov Decision Process including all its parameters, current implementation is adapted
    to Taxi-v3
    """

    def __init__(self, mdp_env, gamma=1.0):
        self.env = mdp_env
        self.gamma = gamma
        self.S = self.env
        self.A = list(range(NUM_ACTIONS))

    def P(self, next_s, curr_s, a):
        if (self.env[curr_s][a][1] == next_s) and (self.env[curr_s][a][3] is False):
            return self.env[curr_s][a][0]
        else:
            return 0

    def R(self, curr_s, a):
        return self.env[curr_s][a][2]


def learn_env_params_taxi():
    """
    Learn the Taxi-v3 environment parameters by running all actions in the environment
    :return: an MDP object with the parameters learned from the environment
    """

    env = gym.make('Taxi-v3')
    env.reset()

    taxi_mdp = {
        state: {action: [] for action in range(NUM_ACTIONS)}
        for state in range(NUM_STATES)
    }

    #  Go over all states
    for i in range(NUM_STATES):
        env.env.s = i

        # Go over all actions for each relevant state
        for j in range(NUM_ACTIONS):
            observation, reward, done, info = env.step(j)
            taxi_mdp[i][j] = [PROBABILITY, observation, reward, done]

            # Reset the environment to the previous state
            env.reset()
            env.env.s = i

    return MarkovDecisionProcess(taxi_mdp, gamma=GAMMA)


def policy_evaluation(policy, env_mdp):
    """
    Policy evaluation algorithm
    :param policy: The policy to evaluate
    :param env_mdp: The MDP and MRP of the environment
    :return: The policy's V array (i.e. value function for each state of the optimal policy)
    """

    # Make a guess on V(s) for all s
    curr_V = {s: env_mdp.A[0] for s in env_mdp.S}

    while True:

        # Keep copy of the V(s) calculated so far
        prev_V = curr_V.copy()

        # Calculate the next V(s) for all states using the state-value function of this policy
        for curr_s in env_mdp.S:
            a = policy[curr_s]

            # No need to loop through all next states as this is deterministic, i.e. given curr_s and action a we know
            # exactly the next state (next_s). The env_mdp.P function is only used for final states that return 0,
            # otherwise it will never converge
            next_s = env_mdp.S[curr_s][a][1]
            curr_V[curr_s] = env_mdp.R(curr_s, a) + env_mdp.P(next_s, curr_s, a) * prev_V[next_s] * env_mdp.gamma

        # Stop on convergence
        if all(np.abs(prev_V[s] - curr_V[s]) <= EPSILON for s in env_mdp.S):
            break

    return curr_V


def policy_improvement(V, env_mdp):
    """
    Greedy policy improvement algorithm
    :param V: The current policy's V array (i.e. value function for each state of the optimal policy)
    :param env_mdp: The MDP and MRP of the environment
    :return: The improved policy
    """

    curr_policy = {s: env_mdp.A[0] for s in env_mdp.S}

    for s in env_mdp.S:
        Q = {}

        for a in env_mdp.A:
            Q[a] = env_mdp.R(s, a) + sum(env_mdp.P(next_s, s, a) * V[next_s] * env_mdp.gamma for next_s in env_mdp.S)

        curr_policy[s] = max(Q, key=Q.get)

    return curr_policy


def policy_iteration(env_mdp):
    """
    The policy iteration algorithm
    :param env_mdp: The MDP and MRP of the environment
    :return: The optimal policy found, and it's V array (i.e. value function for each state of the optimal policy)
    """

    # Random initial policy
    rng = np.random.default_rng()
    curr_policy = {s: rng.integers(0, high=len(env_mdp.A)) for s in env_mdp.S}

    policy_trial_reward = []

    i = 0
    while True:

        prev_policy = curr_policy.copy()

        # Evaluate the current policy, then improve it
        V = policy_evaluation(curr_policy, env_mdp)

        curr_policy = policy_improvement(V, env_mdp)

        # Test the current policy on the game for 50 attempts and record the result
        mean_reward = simulate_policy(curr_policy, num_trials=TEST_TRIALS)
        policy_trial_reward.append(mean_reward)

        print(f"Iteration {i}, policy average total reward: {mean_reward}")

        i += 1

        if all(prev_policy[s] == curr_policy[s] for s in env_mdp.S):
            break
        else:
            prev_set = set(prev_policy.items())
            curr_set = set(curr_policy.items())
            print(f"{len(dict(prev_set ^ curr_set))} policy states changed")

    plot_policy_iteration(policy_trial_reward)

    return curr_policy, V


def plot_policy_iteration(init_state_values):
    """
    Plot the policy iteration convergence diagram
    :param init_state_values: A list with the mean reward of X simulations per step
    """

    plt.plot(np.arange(0, len(init_state_values)), init_state_values)
    plt.xlabel('Improvement iterations')
    plt.ylabel(f'Policy value over {TEST_TRIALS} simulations')

    plt.show()


def simulate_policy(policy, num_trials=10, verbose=False, render=False):
    """
    Simulate a given policy on Taxi-v3 game
    :param policy: the policy to simulate
    :param num_trials: number of simulation games
    :param verbose: Verbose output per step
    :param render: Rendering per step
    :return: The mean reward of all games
    """

    env = gym.make('Taxi-v3')
    mean_reward = 0
    for i in range(num_trials):

        env.reset()
        if render:
            env.render()
        total_reward = 0
        while True:

            curr_state_str = env.decode(env.s)
            next_step = policy[env.s]
            observation, reward, done, info = env.step(next_step)

            if verbose:
                print(f"{list(curr_state_str)}, {STATE_DICT[next_step]}, {reward} \\newline")
            if render:
                env.render()

            total_reward += reward
            if done:
                mean_reward += total_reward
                if verbose:
                    print(f'Total reward for experiment {i}: {total_reward}')
                break

    mean_reward /= num_trials
    return mean_reward


def modify_env(env):

    def new_reset(state=None):
        env.orig_reset()
        if state is not None:
            env.env.s = state
        return np.array(env.env.s)

    env.orig_reset = env.reset
    env.reset = new_reset
    return env


if __name__ == '__main__':

    # Learn the Taxi environment parameters (transition and rewards MDP)
    taxi_mdp = learn_env_params_taxi()

    # Calculate best policy using policy iteration algorithm and keep it in a file (no need to recalculate every time)
    if not os.path.isfile(BEST_POLICY_FILE):
        best_policy, check_states = policy_iteration(taxi_mdp)
        # with open(BEST_POLICY_FILE, 'wb') as file:
        #    pkl.dump([best_policy, check_states], file)
        #    print(f'Optimal policy was calculated and saved at {BEST_POLICY_FILE}')
    else:
        with open(BEST_POLICY_FILE, 'rb') as file:
            best_policy, check_states = pkl.load(file)
            print(f'Optimal policy was already calculated and loaded from {BEST_POLICY_FILE}')

    # Run three simulations as required (1)
    print("\nThree simulation runs for PDF:")
    simulate_policy(best_policy, num_trials=3, verbose=True)

    # Run one simulation with rendering (2)
    print("\nOne simulation run with rendering:")
    simulate_policy(best_policy, num_trials=1, verbose=False, render=True)

    # Print the check states array (3)
    print(f"\ncheck_states array:\n{check_states}")
