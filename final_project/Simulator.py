import DataPreparation
import numpy as np

class Simulator:

    def __init__(self, stock_symbol, start_date, end_date, interval):
        self.dataSet = DataPreparation.DataPreparation().get_dataset(stock_symbol, start_date, end_date, interval)
        self.current_state_index = 0
        self.rng = np.random.default_rng()

    def step(self, action):

        reward = DataPreparation.DataPreparation().calculate_reward_for_state(self.dataSet[self.current_state_index],
                                                                              action,
                                                                              self.dataSet[self.current_state_index + 1])
        self.current_state_index += 1
        # If the next state we got to is the final, make sure to return done (i.e. this transition is the last one)
        if self.current_state_index == len(self.dataSet) - 1:
            done = True
        else:
            done = False

        return self.current_state_index, reward, done

    def reset(self):

        self.current_state_index = 0
        return self.dataSet[self.current_state_index]

    def sample_action(self):

        # Return an integer out of {-1,0,1}
        return self.rng.integers(low=-1, high=2)

