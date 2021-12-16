import gym
import numpy as np
from itertools import product

POSITION_GAUSS_CENTERS = 4
VELOCITY_GAUSS_CENTERS = 8
NUM_WEIGHTS_PER_ACTION = POSITION_GAUSS_CENTERS * VELOCITY_GAUSS_CENTERS
COV_MATRIX_INV = np.linalg.inv(np.diag([0.04, 0.0004]))
CENTERS = np.zeros((2, POSITION_GAUSS_CENTERS * VELOCITY_GAUSS_CENTERS))

MAX_STEPS_PER_EPISODE = 200
MAX_TOTAL_STEPS = 1e6
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
    rng = np.random.default_rng()

    c_p = rng.uniform(p_range[0], p_range[1], POSITION_GAUSS_CENTERS)
    c_v = rng.uniform(v_range[0], v_range[1], VELOCITY_GAUSS_CENTERS)

    print(f'Position centers: {c_p}\nVelocity centers: {c_v}')
    return np.transpose(np.array(list(product(c_p, c_v))))


def generate_uniform_centers(p_range, v_range):
    p_step = (p_range[1] - p_range[0]) / POSITION_GAUSS_CENTERS
    v_step = (v_range[1] - v_range[0]) / VELOCITY_GAUSS_CENTERS
    c_p = np.arange(p_range[0], p_range[1], p_step)
    c_v = np.arange(v_range[0], v_range[1], v_step)

    print(f'Position centers: {c_p}\nVelocity centers: {c_v}')
    return np.transpose(np.array(list(product(c_p, c_v))))


def state_to_feature_vector(s):
    x = s[..., np.newaxis] - CENTERS
    return np.diag(np.exp(-0.5 * (np.transpose(x) @ COV_MATRIX_INV @ x)))


def Q(s, a, w):
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
    if rng.uniform(0, 1) < epsilon:
        return env.action_space.sample() # exploration
    else:
        return choose_best_action(current_state, w)

def control_algorithm(environment, gamma=1, alpha=0.02, lambda_val=0.5, epsilon=0.995, lambda_value=0,
                      verbose=False, plot=False):
    # Random init
    rng = np.random.default_rng()

    # Init weights
    n_actions = environment.action_space.n
    w = np.zeros((n_actions, NUM_WEIGHTS_PER_ACTION))

    total_steps = 0
    num_episodes = 0
    tests_counter = 1
    curr_epsilon = epsilon
    init_state_values = []
    max_avg_reward = 0
    best_w = None
    last_E = 0

    while total_steps < MAX_TOTAL_STEPS:

        curr_state = environment.reset()
        total_rewards = 0
        for i in range(MAX_STEPS_PER_EPISODE):
            # Choose next action according to epsilon-greedy
            action = epsilon_greedy_next_action(env, rng, curr_epsilon, Q, curr_state, w)

            # Action
            next_state, reward, done, prob = environment.step(action)
            total_steps += 1

            # Calculate delta
            old_q_val = Q(curr_state, action, w)
            # Q learning
            #delta = (reward + gamma * np.max(
            #    [Q(next_state, 0, w), Q(next_state, 1, w), Q(next_state, 2, w)])) - old_q_val

            # Backward-view TD lambda
            next_action = epsilon_greedy_next_action(env, rng, curr_epsilon, Q, next_state, w)
            delta = reward + gamma * Q(next_state, next_action, w) - old_q_val
            curr_E = gamma * lambda_val * last_E + state_to_feature_vector(curr_state)

            # SGD step
            #w[action] += alpha * delta * state_to_feature_vector(curr_state) # Q-learning
            w[action] += alpha * delta * curr_E # Backward-view TD lambda

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
            init_state_values.append([STEPS_FOR_TEST * tests_counter, mean_reward])

            # Store the policy which scored best mean reward (as the convergence process is rather noisy)
            if mean_reward > max_avg_reward:
                max_avg_reward = mean_reward
                best_w = w
            print(f'mean reward = {mean_reward}, epsilon = {curr_epsilon}')

        # Update exploration with exponential decay
        # curr_epsilon = epsilon * np.exp(-EPSILON_DECAY * num_episodes)
        curr_epsilon = np.max([epsilon - EPSILON_DECAY * total_steps, MIN_EPSILON])

    print(f'Sarsa Lambda finished, total steps = {total_steps}, mean reaward for best policy = {max_avg_reward}')

    return best_w


def simulate_policy(env, w, num_trials=NUM_SIMULATIONS, gamma=0.95, verbose=False, render=False, for_latex=False):
    total_rewards = 0

    for i in range(num_trials):
        ep_reward = 0
        steps = 0
        curr_state = env.reset()

        while True:
            action = choose_best_action(curr_state, w)

            # Action
            next_state, reward, done, prob = env.step(action)
            steps += 1
            ep_reward += reward # * (gamma ** steps)  

            curr_state = next_state

            if done:
                total_rewards += ep_reward
                #print(f'Episode finished with reward={reward} at the last {steps} step, agent pos = {next_state}')
                break

    mean_reward = total_rewards / num_trials
    return mean_reward


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')

    CENTERS = generate_uniform_centers((env.env.min_position, env.env.max_position), (-env.env.max_speed, env.env.max_speed))

    control_algorithm(env)
