import numpy as np
import torch

import LSTM
import Simulator
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


INPUT_SIZE = 7
HIDDEN_SIZE = 64
CONVERGENCE_DELTA = 1e5
MEM_SIZE = 5000
BATCH_SIZE = 64
GAMMA = 0.3
ALPHA = 0.0001
TAU = 1000
EPSILON = 0.3
EPSILON_DECAY_RATE = 0.0001


class DQN:

    def __init__(self, env: Simulator.Simulator, epsilon, gamma=GAMMA):

        self.epsilon = epsilon
        self.gamma = gamma
        self.rng = np.random.default_rng()
        self.env = env.reset()
        self.replay_buffer = ReplayMemory(MEM_SIZE)

        # Initializing both online and target NNs with random weights
        self.online_network = LSTM.LSTMAutoEncoder(INPUT_SIZE, HIDDEN_SIZE)
        self.target_network = LSTM.LSTMAutoEncoder(INPUT_SIZE, HIDDEN_SIZE)


    def choose_act_eps_greedy(self, curr_state):

        # Sample phase, choose next step according to epsilon-greedy
        if self.rng.uniform(0, 1) < self.epsilon:
            action = self.env.sample_action()  # exploration
        else:
            action = np.argmax(self.online_network.predict_Q(curr_state))  # greedy
        return action

    def execute_DQN_algorithm(self, epsilon, gamma=GAMMA):

        steps = 0
        episodes = 0
        loss_list = []
        done = False

        while True:

            # Start a new episode
            curr_state = self.env.reset()

            while not done:

                # Choose action using epsilon greedy and take it
                action = self.choose_act_eps_greedy(curr_state)
                next_state, reward, done = self.env.step(action)
                steps += 1

                self.replay_buffer.add_transition(np.array([curr_state, action, reward, next_state, done, 0]))

                if self.replay_buffer.size() >= BATCH_SIZE:

                    # Sample a batch
                    batch = self.replay_buffer.sample(BATCH_SIZE)

                    for transition in batch:
                        # Determine the label for this action and fill it in the transition
                        if transition[4]: # if DONE
                            label = reward
                        else:
                            max_action_chosen = np.argmax(self.online_network.predict_Q(next_state))
                            label = reward + gamma * self.target_network.predict_Q(max_action_chosen)
                        transition[5] = label

                    loss = LSTM.train(self.online_network, torch.device, batch, ALPHA)
                    loss_list.append([steps, episodes, next_loss])

                    # Copy weights from online network to target network every tau steps
                    if steps % TAU == 0:
                        self.target_network.load_state_dict(self.online_network.state_dict()) #

                    # Epsilon decay
                    self.epsilon = self.epsilon * (1 - EPSILON_DECAY_RATE)

            # Episode is done here
            episodes += 1

            # Stop training when we reach convergence
            if len(loss_list) >= 2:
                if np.abs(loss_list[-1][2] - loss_list[-2][2]) < CONVERGENCE_DELTA:
                    break

        return loss_list


class ReplayMemory:

    def __init__(self, mem_size):

        self.max_mem_size = mem_size
        self.filled_mem_index = 0
        self.mem = np.zeros(mem_size, 4)
        self.purge_index = 0

    def sample(self, batch_size):

        max_sapmle_idx = self.filled_mem_index
        batch_indices = np.random.choice(max_sapmle_idx, batch_size, replace=False)

        return [self.mem[batch_indices]]

    def add_transition(self, transition):

        if self.filled_mem_index < self.max_mem_size - 1: # If memory isn't full yet, just add in the last cell
            self.mem[self.filled_mem_index] = transition
            self.filled_mem_index += 1
        elif self.purge_index < self.max_mem_size - 1: # If memory is full, override the oldest entry (purge index)
            self.mem[self.purge_index] = transition
            self.purge_index += 1
        else:  # If memory is full and also the entire memory was already purged
            self.mem[0] = transition
            self.purge_index = 1

    def size(self):
        return self.filled_mem_index


