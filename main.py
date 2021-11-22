import gym
import numpy as np

import assignment_01.assignment_01 as ex1





if __name__ == '__main__':

    transition_matrix, reward_matrix = ex1.learn_env_params_taxi()

    # env = gym.make('Taxi-v3')
    # observation = env.reset()
    #
    # for _ in range(200):
    #     env.render(mode='human')
    #
    #     msg = '\n- 0: move south\n- 1: move north\n- 2: move east\n- 3: move west\n- 4: pickup passenger\n- 5: drop off passenger\n'
    #     action = int(input(msg))
    #     observation, reward, done, info = env.step(action)
    #     print(f"observation: {observation}, reward:{reward}, done:{done}, info:{info}")
    #     if done:
    #         observation = env.reset()
    #
    #     env.close()





# See PyCharm help at https://www.jetbrains.com/help/pycharm/
