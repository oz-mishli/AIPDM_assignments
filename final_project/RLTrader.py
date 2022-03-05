import numpy as np
from LSTM import LSTMNetwork


class RLTrader:
    ACTION_NAME = {-1: 'short', 0: 'do nothing', 1: 'call'}

    def __init__(self, model: LSTMNetwork, test_states):
        self.current_roi = 1
        self.test_states = test_states
        self.model = model
        self.action_count = []
        self.prices = []
        self.has_open_call = None
        self.has_open_short = None

    def run_simulation(self) -> float:
        for index in range(self.test_states.shape[0] - 1):
            current_state = self.test_states[index]
            current_price = current_state[-1, 0]
            next_action = np.argmax(self.model.forward(current_state)) - 1
            self._perform_next_action(next_action, current_price)
            print(f'price={current_price}, bot decided to {self.ACTION_NAME[next_action]}')

    def _next_action_validation(self, next_action):
        if next_action == -1 and self.has_open_short:
            return 0
        if next_action == 1 and self.has_open_call:
            return 0
        return next_action

    def _perform_next_action(self, next_action, current_price):
        validated_action = self._next_action_validation(next_action)
        self.action_count.append(validated_action)
        self.prices.append(current_price)
        self._sell_holdings(next_action, current_price)

        if validated_action == 1 and not self.has_open_call:
            self.has_open_call = current_price
            self.has_open_short = None
        elif validated_action == -1 and not self.has_open_short:
            self.has_open_short = current_price
            self.has_open_call = None

    def _sell_holdings(self, validated_action, current_price):
        if validated_action == 1 and self.has_open_short:
            self.current_roi *= (self.has_open_short - current_price) / self.has_open_short + 1
        if validated_action == -1 and self.has_open_call:
            self.current_roi *= (current_price - self.has_open_call) / self.has_open_call + 1
