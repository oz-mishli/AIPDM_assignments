from itertools import product
import numpy as np
import gym
import time

NUM_DISCRETE_ACTIONS_PER_RANGE = 3
ACTIONS_MAPPING = dict()
ANGLE_GAUSS_CENTERS_NUM = 4
VELOCITY_GAUSS_CENTERS_NUM = 8
ACTION_GAUSS_CENTERS_NUM = 3
MAX_STEPS = 1e6
NUM_SIMULATIONS = 100
STEPS_FOR_TEST = 1000

# Actor centers are only for state
ACTOR_CENTERS = np.zeros((2, ANGLE_GAUSS_CENTERS_NUM * VELOCITY_GAUSS_CENTERS_NUM))

# Critic centers are both for action and state
CRITIC_CENTERS = np.zeros((3, ANGLE_GAUSS_CENTERS_NUM * VELOCITY_GAUSS_CENTERS_NUM * ACTION_GAUSS_CENTERS_NUM))

ACTOR_COV_MATRIX_INV = np.linalg.inv(np.diag([0.04, 0.0004]))
CRITIC_COV_MATRIX_INV = np.linalg.inv(np.diag([0.04, 0.0004, 0.0001]))  # TODO have no idea which values to put here...
VARIANCE = 0.5  # TODO find good value for variance hyper-parameter
ALPHA = 0.001
BETA = 0.01
GAMMA = 1


def discretize_action_space(env):
    min = env.action_space.low
    max = env.action_space.high

    actions = np.arange(min, 0, (max - min) / (NUM_DISCRETE_ACTIONS_PER_RANGE * 2), dtype=float)

    actions_mapping = {
        0: (actions[0], 'la-3'),
        1: (actions[1], 'la-2'),
        2: (actions[2], 'la-1'),
        3: (0.0, 'na'),
        4: (-actions[2], 'ra-1'),
        5: (-actions[1], 'ra-2'),
        6: (-actions[0], 'ra-3')
    }
    return actions_mapping


def actor_critic(env, alpha=ALPHA, beta=BETA, gamma=GAMMA):
    theta = np.zeros(ANGLE_GAUSS_CENTERS_NUM * VELOCITY_GAUSS_CENTERS_NUM)
    w = np.zeros(ANGLE_GAUSS_CENTERS_NUM * VELOCITY_GAUSS_CENTERS_NUM * ACTION_GAUSS_CENTERS_NUM)
    env.reset()
    s = env.state
    total_steps = 0
    tests_counter = 1
    simulation_series = list()

    a = sample_from_gaussian_policy(s, theta)

    while total_steps < MAX_STEPS:
        next_state, reward, done, prob = env.step([a])
        next_state = env.state
        next_action = sample_from_gaussian_policy(next_state, theta)

        old_Q = Q_critic(s, a, w)
        delta = reward + gamma * Q_critic(next_state, next_action, w) - old_Q
        theta += alpha * gaussian_score_function(s, a, theta) * old_Q
        w += beta * delta * state_action_to_feature_vector(s, a)

        a = next_action
        s = next_state
        total_steps += 1

        # Simulate current policy
        if total_steps > STEPS_FOR_TEST * tests_counter:
            tests_counter += 1
            mean_reward = simulate_policy(env, theta)
            simulation_series.append((STEPS_FOR_TEST * tests_counter, mean_reward))

            print(f'mean reward = {mean_reward}')


def simulate_policy(env, theta, num_trials=NUM_SIMULATIONS, verbose=False, render=False, for_latex=False):
    total_rewards = 0

    for i in range(num_trials):
        ep_reward = 0
        steps = 0
        env.reset()
        curr_state = env.state

        while True:
            if render:
                env.render()
                time.sleep(0.025)  # Add delay so we can meaningfully watch the episode
            action = sample_from_gaussian_policy(curr_state, theta)

            # Action
            next_state, reward, done, prob = env.step([action])
            next_state = env.state
            steps += 1
            ep_reward += reward

            # if verbose:
            #     if for_latex:
            #         print(f'\\item{{}} {next_state[0]}, {next_state[1]}, 0.5, 0, {ACTION_NAME_MAPPING[action]}, {reward}')
            #     else:
            #         print(f'{steps}. {next_state[0]}, {next_state[1]}, 0.5, 0, {ACTION_NAME_MAPPING[action]}, {reward}')

            curr_state = next_state

            if done:
                total_rewards += ep_reward
                if verbose:
                    print(f'Trial episode {i} finished with reward {ep_reward}')
                break

    mean_reward = total_rewards / num_trials
    return mean_reward


def generate_actor_centers(a_range, v_range):
    """
    Generate fixed centers for the RBF features, evenly distributed within the respective ranges for position and
    velocity
    :param p_range: tuple of (a, b) representing the range of positions available
    :param v_range: tuple if (a, b) representing the range of speed values
    :return: Numpy array with all the circle centers of shape (2, POSITION_GAUSS_CENTERS * VELOCITY_GAUSS_CENTERS) This
    array contains all possible combinations of position circles and velocity circles (i.e. cartersian product)
    """

    p_step = (a_range[1] - a_range[0]) / ANGLE_GAUSS_CENTERS_NUM
    v_step = (v_range[1] - v_range[0]) / VELOCITY_GAUSS_CENTERS_NUM
    c_p = np.arange(a_range[0], a_range[1], p_step)
    c_v = np.arange(v_range[0], v_range[1], v_step)

    print(f'Angle centers: {c_p}\nVelocity centers: {c_v}')
    return np.transpose(np.array(list(product(c_p, c_v))))


def generate_critic_centers(a_range, v_range, act_range):
    p_step = (a_range[1] - a_range[0]) / ANGLE_GAUSS_CENTERS_NUM
    v_step = (v_range[1] - v_range[0]) / VELOCITY_GAUSS_CENTERS_NUM
    act_step = (act_range[1] - act_range[0]) / ACTION_GAUSS_CENTERS_NUM
    c_p = np.arange(a_range[0], a_range[1], p_step)
    c_v = np.arange(v_range[0], v_range[1], v_step)
    c_act = np.arange(act_range[0], act_range[1], act_step)

    print(f'Angle centers: {c_p}\nVelocity centers: {c_v}\nAction centers: {c_act}')
    return np.transpose(np.array(list(product(c_p, c_v, c_act))))


def state_to_feature_vector(s):
    """
    Translate from a given state to a feature vector representing that state
    :param s: Input state
    :return: The feature vector representing the input state
    """
    x = s[..., np.newaxis] - ACTOR_CENTERS
    return np.diag(np.exp(-0.5 * (np.transpose(x) @ ACTOR_COV_MATRIX_INV @ x)))


def state_action_to_feature_vector(s, a):
    """

    :param s:
    :param a:
    :return:
    """

    x = np.append(s, a)  # add the action as the third dimension
    x = x[..., np.newaxis] - CRITIC_CENTERS
    return np.diag(np.exp(-0.5 * np.transpose(x) @ CRITIC_COV_MATRIX_INV @ x))


def sample_from_gaussian_policy(s, theta):
    mean = np.dot(state_to_feature_vector(s), theta)
    return np.random.normal(loc=mean, scale=VARIANCE)


def gaussian_score_function(s, a, theta):
    mean = np.dot(state_to_feature_vector(s), theta)
    factor = (a - mean) / VARIANCE ** 2

    return factor * state_to_feature_vector(s)


def Q_critic(s, a, w):
    theta = state_action_to_feature_vector(s, a)
    return np.dot(theta, w)


if __name__ == '__main__':
    env = gym.make('Pendulum-v1')

    ACTOR_CENTERS = generate_actor_centers((-np.pi, np.pi),
                                           (-env.env.max_speed, env.env.max_speed))
    CRITIC_CENTERS = generate_critic_centers((-np.pi, np.pi),
                                             (-env.env.max_speed, env.env.max_speed),
                                             (env.action_space.low, env.action_space.high))

    actor_critic(env)

    # Draft code for analyzing the env
    # sp = env.observation_space
    #
    # start = env.reset()
    # print(start)
    #
    # env.render()
    # steps = 0
    # while(steps < 100):
    #     step = env.step([2])
    #     print(step)
    #     print(env.state)
    #     env.render()
    #     time.sleep(1)
    #
    #
    #
    #
    #
    # x = env.step(3)
