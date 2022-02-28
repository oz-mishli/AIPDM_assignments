import DataPreparation
import numpy as np
import pickle as pkl
import os.path


class Simulator:

    def __init__(self, ds_fname, stock_symbol, start_date, end_date, interval):
        self.load_dataset(ds_fname, stock_symbol, start_date, end_date, interval)
        self.current_state_index = 0
        self.rng = np.random.default_rng()

    def load_dataset(self, file_name, stock_symbol, start_date, end_date, interval):

        if os.path.isfile(file_name):
            with open(file_name, 'rb') as ds_file:
                out_ds = pkl.load(ds_file)
        else:
            out_ds = DataPreparation.DataPreparation().get_dataset(stock_symbol, start_date, end_date, interval)
            with open(file_name, 'wb') as ds_file:
                pkl.dump(out_ds, ds_file)

        self.dataSet = out_ds

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

        return np.array(self.dataSet[self.current_state_index], dtype=np.float), reward, done

    def reset(self):

        self.current_state_index = 0

        # TODO: convert to float may be more efficient at the source when building the dataset
        return np.array(self.dataSet[self.current_state_index], dtype=np.float)

    def sample_action(self):

        # Return an integer out of {-1,0,1}
        return self.rng.integers(low=-1, high=2)

