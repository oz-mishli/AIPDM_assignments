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


class DQN:

    def __init__(self):
        return 0


    def execute_DQN_algorithm(self, epsilon, gamma=GAMMA):

        # Initializing both online and target NNs with random weights
        online_network = LSTM.LSTMAutoEncoder(INPUT_SIZE, HIDDEN_SIZE)
        target_network = LSTM.LSTMAutoEncoder(INPUT_SIZE, HIDDEN_SIZE)

        # Init replay memory
        replay_mem = []

        rng = np.random.default_rng()
        # Init env
        env = Simulator.Simulator()
        curr_state = env.reset()

        loss = 0
        next_loss = 1
        curr_epsilon = epsilon

        while np.abs(loss - next_loss) > CONVERGENCE_DELTA:

            # Sample phase, choose next step according to epsilon-greedy
            if rng.uniform(0, 1) < curr_epsilon:
                action = env.sample_action()  # exploration
            else:
                action = np.argmax(online_network.predict_Q(curr_state)) # greedy

            # Take chose action
            next_state, reward, done = env.step(action)

            # Determine the label for this action
            if done:
                label = reward
            else:
                max_action_chosen = np.argmax(online_network.predict_Q(next_state))
                label = reward + gamma * target_network.predict_Q(max_action_chosen)

            replay_mem.append([next_state, reward, done, label])

            if (len(replay_mem) >= MEM_SIZE):
                # Load the replay memory data in a dataset
                td = TransitionsDataset(replay_mem)
                td_loader = DataLoader(td, batch_size=BATCH_SIZE, shuffle=True)
















        return 0


class TransitionsDataset(Dataset):
    """
    A Pytorch Dataset class for the transitions dataset we collect during training
    """

    def __init__(self, transitions_list):
        self.transitions = torch.FloatTensor(transitions_list)

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        return self.transitions[idx][0:3], self.transitions[idx][3]
