import numpy as np
import LSTM
import Simulator

INPUT_SIZE = 7
HIDDEN_SIZE = 64
CONVERGENCE_DELTA = 1e5
MEM_SIZE = 5000


class DQN:

    def __init__(self):
        return 0


    def execute_DQN_algorithm(self, epsilon):

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

        while (np.abs(loss - next_loss) > CONVERGENCE_DELTA):

            # Sample phase, choose next step according to epsilon-greedy
            if rng.uniform(0, 1) < curr_epsilon:
                action = env.sample_action()  # exploration
            else:
                action = np.argmax(online_network.predict_Q(curr_state)) # greedy

            # Take chose action
            next_state, reward, done = env.step(action)
            replay_mem.append([next_state, reward, done])

            #if (len(replay_mem) >= MEM_SIZE):












        return 0