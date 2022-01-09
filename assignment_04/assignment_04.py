#import keyboard For env simulation, not used
#import pynput.keyboard  For env simulation, not used
#from pynput.keyboard import Key, Listener  For env simulation, not used

from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import gym
import time



NUM_DISCRETE_ACTIONS_PER_RANGE = 3
ACTIONS_MAPPING = dict()
ANGLE_GAUSS_CENTERS_NUM = 4
VELOCITY_GAUSS_CENTERS_NUM = 8
ACTION_GAUSS_CENTERS_NUM = 3
MAX_STEPS = 1e6
MAX_EPISODE_STEPS = 200
NUM_SIMULATIONS = 100
STEPS_FOR_TEST = 10000
REWARD_NORM_CONSTANT = 15

# Actor centers are only for state
ACTOR_CENTERS = np.zeros((2, ANGLE_GAUSS_CENTERS_NUM * VELOCITY_GAUSS_CENTERS_NUM))

# Critic centers are both for action and state
CRITIC_CENTERS = np.zeros((3, ANGLE_GAUSS_CENTERS_NUM * VELOCITY_GAUSS_CENTERS_NUM * ACTION_GAUSS_CENTERS_NUM))

ACTOR_COV_MATRIX_INV = np.linalg.inv(np.diag([0.04, 0.0004]))
CRITIC_COV_MATRIX_INV = np.linalg.inv(np.diag([0.04, 0.0004, 0.0001]))
VARIANCE = 10
ALPHA = 0.001
BETA = 0.01
GAMMA = 1


def actor_critic(env, alpha=ALPHA, beta=BETA, gamma=GAMMA, plot=False):

    rng = np.random.default_rng()

    theta = np.random.random((ANGLE_GAUSS_CENTERS_NUM * VELOCITY_GAUSS_CENTERS_NUM))
    w = np.zeros(ANGLE_GAUSS_CENTERS_NUM * VELOCITY_GAUSS_CENTERS_NUM * ACTION_GAUSS_CENTERS_NUM)
    env.reset()
    s = env.state
    total_steps = 0
    tests_counter = 1
    simulation_series = list()
    simulation_vals = list()
    done = False
    ep_reward = 0
    best_theta = 0
    best_mean_reward = 0

    a, un_normalized_a = sample_from_gaussian_policy(s, theta)

    while total_steps < MAX_STEPS:

        while not done:
            next_state, reward, done, prob = env.step([a])
            ep_reward += reward
            if done:
                env.reset()
                s = env.state
                ep_reward = 0
                ep_steps = 0
                done = False
                break

            reward += REWARD_NORM_CONSTANT
            next_state = env.state
            next_action, next_action_un_normalized = sample_from_gaussian_policy(next_state, theta)

            old_Q = Q_critic(s, a, w)
            delta = reward + gamma * Q_critic(next_state, next_action, w) - old_Q

            # Actor update
            theta += alpha * gaussian_score_function(s, a, theta) * old_Q

            # Critic update
            w += beta * delta * state_action_to_feature_vector(s, a)

            a = next_action
            s = next_state
            total_steps += 1

        # Simulate current policy
        if total_steps > STEPS_FOR_TEST * tests_counter:
            tests_counter += 1
            mean_reward = simulate_policy(env, theta)
            if mean_reward > best_mean_reward:
                best_theta = theta
                best_mean_reward = mean_reward
            simulation_series.append(STEPS_FOR_TEST * tests_counter)
            simulation_vals.append(mean_reward)


            print(f'mean reward = {mean_reward}')

    if plot:
        plt.plot(simulation_series, simulation_vals)
        plt.xlabel('Control Steps')
        plt.ylabel('Average Mean Reward')
        plt.show()

    return best_theta


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
            action, un_norm_action = sample_from_gaussian_policy(curr_state, theta)

            # Action
            next_state, reward, done, prob = env.step([action])
            reward += REWARD_NORM_CONSTANT

            next_state = env.state
            steps += 1
            ep_reward += reward

            if verbose:
                if for_latex:
                    print(f'\\item{{}} theta: {next_state[0]}, theta dot: {next_state[1]}, action: {action}, reward:{reward}')
                else:
                     print(f'{steps}. theta: {next_state[0]}, theta dot: {next_state[1]}, action: {action}, reward:{reward}')

            curr_state = next_state

            if (done) or (steps > MAX_EPISODE_STEPS):
                total_rewards += ep_reward
                if verbose:
                    print(f'Trial episode {i} finished with reward {ep_reward}')
                break

    mean_reward = total_rewards / num_trials
    return mean_reward


def generate_actor_centers(a_range, v_range):
    """
    Generate fixed centers for the actor RBF features, evenly distributed within the respective ranges for position and
    velocity
    :param a_range: tuple of (a, b) representing the range of angles available
    :param v_range: tuple of (a, b) representing the range of velocity values
    :return: Numpy array with all the circle centers of shape (2, POSITION_GAUSS_CENTERS * VELOCITY_GAUSS_CENTERS) This
    array contains all possible combinations of position circles and velocity circles (i.e. cartesian product)
    """

    p_step = (a_range[1] - a_range[0]) / ANGLE_GAUSS_CENTERS_NUM
    v_step = (v_range[1] - v_range[0]) / VELOCITY_GAUSS_CENTERS_NUM
    c_p = np.arange(a_range[0], a_range[1], p_step)
    c_v = np.arange(v_range[0], v_range[1], v_step)

    print(f'Angle centers: {c_p}\nVelocity centers: {c_v}')
    return np.transpose(np.array(list(product(c_p, c_v))))


def generate_critic_centers(a_range, v_range, act_range):
    """
    Generate fixed centers for the critic RBF features, evenly distributed within the respective ranges for position,
    velocity and action
    :param a_range: tuple of (a, b) representing the range of angles available
    :param v_range: tuple of (a, b) representing the range of velocity values
    :param act_range: tuple of (a, b) representing the range of action values
    :return:
    """
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
    Translate from a given state and action to a feature vector representing that state and action
    :param s: Input state
    :param a: Input action
    :return: The feature vector representing the input state and input action
    """

    x = np.append(s, a)  # add the action as the third dimension
    x = x[..., np.newaxis] - CRITIC_CENTERS
    return np.diag(np.exp(-0.5 * np.transpose(x) @ CRITIC_COV_MATRIX_INV @ x))


def sample_from_gaussian_policy(s, theta):
    """
    Sample an action from the Gaussian policy
    :param s: the current state
    :param theta: the parameters of the policy
    :return:
    """
    mean = np.dot(state_to_feature_vector(s), theta)
    orig_val = np.random.normal(loc=mean, scale=VARIANCE)
    normalized_val = np.tanh(orig_val) * 2

    # Return the normalized value of the policy, as well as the original value (Implemented according to section 2 of
    # Ronen's note)
    return normalized_val, orig_val


def gaussian_score_function(s, a, theta):
    """
    Calculate the Gaussian score function
    :param s: input state
    :param a: input action
    :param theta: current set of parameters for the policy evaluation
    :return: the score of (s, a)
    """
    mean = np.dot(state_to_feature_vector(s), theta)
    factor = (a - mean) / VARIANCE ** 2

    return factor * state_to_feature_vector(s)


def Q_critic(s, a, w):
    theta = state_action_to_feature_vector(s, a)
    return np.dot(theta, w)



#############
# This code is not in use, was used for figuring out the environment manually so we can come up with a hand-crafted
# policy to start with
# env = gym.make('Pendulum-v1')
# def show(key):
#
#     #print('\nYou Entered {0}'.format(key))
#     if key.char == "a":
#         print("a")
#         print(env.step([1]))
#         #env.render()
#     if key.char == "d":
#         print("d")
#         print(env.step([-1]))
#         #env.render()
#
#     #return False
#
#  if key == Key.delete:
#     # Stop listener
#     return False
############


if __name__ == '__main__':

    env = gym.make('Pendulum-v1')


    ACTOR_CENTERS = generate_actor_centers((-np.pi, np.pi),
                                           (-env.env.max_speed, env.env.max_speed))
    CRITIC_CENTERS = generate_critic_centers((-np.pi, np.pi),
                                             (-env.env.max_speed, env.env.max_speed),
                                             (env.action_space.low, env.action_space.high))

    best_theta = actor_critic(env, plot=True)

    print(best_theta)

    simulate_policy(env, best_theta, 1, verbose=True, for_latex=True)

    #############
    # This code is not in use, was used for figuring out the environment manually so we can come up with a hand-crafted
    # policy to start with
    ### Env debugging
    # start = env.reset()
    # env.render()
    # #Draft code for analyzing the env
    # sp = env.observation_space
    #
    # start = env.reset()
    # print(start)
    #
    # # # Collect all event until released
    # # with Listener(on_press=show) as listener:
    # #     listener.start()
    #
    # env.render()
    # steps = 0
    #
    # with pynput.keyboard.Events() as events:
    #
    #
    #     while(steps < 1000):
    #         #step = env.step([2])
    #         #print(step)
    #         #print(env.state)
    #
    #         event = events.get(0.2)
    #         if event is None:
    #             next_state, reward, done, prob = env.step([0])
    #         elif event.key == Key.left:
    #             next_state, reward, done, prob = env.step([-0.5])
    #             print("left")
    #         elif event.key == Key.right:
    #             next_state, reward, done, prob = env.step([0.5])
    #             print("right")
    #         env.render()
    #     #time.sleep(0.1)
    #############

