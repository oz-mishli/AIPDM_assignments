import os.path

import numpy as np
import torch
import pickle as pkl

import LSTM
import Simulator

TRAINING_DS_FNAME = 'train_dataset.pkl'
TEST_DS_FNAME = 'test_dataset.pkl'

INPUT_SIZE = 7
HIDDEN_SIZE = 64
NUM_ACTIONS = 3
CONVERGENCE_DELTA = 1e5
MEM_SIZE = 5000
BATCH_SIZE = 64
GAMMA = 0.3
ALPHA = 0.0001
TAU = 1000
EPSILON = 0.3
EPSILON_DECAY_RATE = 0.0001


class DQN:

    def __init__(self, env: Simulator.Simulator, epsilon, gamma=GAMMA, duelling_network=True):

        self.epsilon = epsilon
        self.gamma = gamma
        self.rng = np.random.default_rng()
        self.env = env
        self.replay_buffer = ReplayMemory(MEM_SIZE)
        self.duelling_network = duelling_network

        # Initializing both online and target NNs with random weights
        self.online_network = LSTM.LSTMNetwork(INPUT_SIZE, HIDDEN_SIZE, NUM_ACTIONS, self.duelling_network)
        self.target_network = LSTM.LSTMNetwork(INPUT_SIZE, HIDDEN_SIZE, NUM_ACTIONS, self.duelling_network)


    def choose_act_eps_greedy(self, curr_state):

        # Sample phase, choose next step according to epsilon-greedy
        if self.rng.uniform(0, 1) < self.epsilon:
            action = self.env.sample_action()  # exploration
        else:
            action = np.argmax(self.online_network.forward(self.convert_state_to_tensor(curr_state)))  # greedy
        return action

    def convert_state_to_tensor(self, s, batch_first=False):

        # Convert the state(s) to a tensor in the matching dimensions before feeding to the LSTM NN
        in_tensor = torch.from_numpy(s)
        if len(in_tensor.shape) == 2:
            if batch_first:
                squeeze_ind = 0
            else:
                squeeze_ind = 1
            in_tensor = torch.unsqueeze(in_tensor, squeeze_ind)
        else:
            in_tensor = s

        return in_tensor.float()



    def execute_DQN_algorithm(self):

        steps = 0
        episodes = 0
        loss_list = []
        done = False
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        while True:

            # Start a new episode
            curr_state = self.env.reset()

            while not done:

                # Choose action using epsilon greedy and take it
                action = self.choose_act_eps_greedy(curr_state)
                next_state, reward, done = self.env.step(action)
                steps += 1

                self.replay_buffer.add_transition([curr_state, action, reward, next_state, done])

                if self.replay_buffer.size() >= BATCH_SIZE:

                    # Sample a batch
                    batch = self.replay_buffer.sample(BATCH_SIZE)
                    targets_for_batch = []
                    #batch_for_model = np.zeros(INPUT_SIZE, self.replay_buffer.size())

                    for transition in batch:
                        # Determine the label for this action and fill it in the transition
                        if transition[4]: # if DONE
                            label = reward
                        else:
                            next_state_tensor = self.convert_state_to_tensor(next_state)
                            max_action_chosen = np.argmax(self.online_network.forward(next_state_tensor))
                            label = reward + self.gamma * np.squeeze(self.target_network.forward(next_state_tensor)[:, max_action_chosen])
                        targets_for_batch.append(label)

                    loss = LSTM.train(self.online_network, device, batch, targets_for_batch, ALPHA)
                    loss_list.append([steps, episodes, loss])
                    print(f'status = {loss_list[-1]}')

                    # Copy weights from online network to target network every tau steps
                    if steps % TAU == 0:
                        self.target_network.load_state_dict(self.online_network.state_dict()) #

                    # Epsilon decay
                    self.epsilon = self.epsilon * (1 - EPSILON_DECAY_RATE)

            # Episode is done here
            episodes += 1
            print('episode is done')

            # Stop training when we reach convergence
            if len(loss_list) >= 2:
                if np.abs(loss_list[-1][2] - loss_list[-2][2]) < CONVERGENCE_DELTA:
                    break

        return loss_list


class ReplayMemory:

    def __init__(self, mem_size):

        self.max_mem_size = mem_size
        self.filled_mem_index = 0
        self.mem = []
        self.purge_index = 0

    def sample(self, batch_size):

        max_sapmle_idx = self.filled_mem_index
        batch_indices = np.random.choice(max_sapmle_idx, batch_size, replace=False)

        return [self.mem[k] for k in batch_indices]

    def add_transition(self, transition):

        if self.filled_mem_index < self.max_mem_size - 1: # If memory isn't full yet, just add in the last cell
            self.mem.append(transition)
            self.filled_mem_index += 1
        elif self.purge_index < self.max_mem_size - 1: # If memory is full, override the oldest entry (purge index)
            self.mem[self.purge_index] = transition
            self.purge_index += 1
        else:  # If memory is full and also the entire memory was already purged
            self.mem[0] = transition
            self.purge_index = 1

    def size(self):
        return self.filled_mem_index


if __name__ == '__main__':

    train_env = Simulator.Simulator(TRAINING_DS_FNAME, 'SPY', '2000-01-03', '2010-12-31', '1d')
    test_env = Simulator.Simulator(TEST_DS_FNAME, 'SPY', '2011-01-03', '2022-02-25', '1d')

    q_net = DQN(train_env, EPSILON, GAMMA, duelling_network=False)
    q_net.execute_DQN_algorithm()










