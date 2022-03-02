from StockDataFetcher import StockDataFetcher
from TechnicalAnalyzer import TechnicalAnalyzer
import numpy as np


class DataPreparation:
    NORMALIZED_PRICE_INDEX = 0
    MU = 0.5
    BP = 0.0001
    STATE_SIZE = 60

    @staticmethod
    def get_dataset(stock_symbol, start_date, end_date, interval):
        """
        Download stock data, calculate its features.
        """
        states = []
        stock_data = StockDataFetcher.fetch_stock_data(stock_symbol, start_date, end_date, interval)
        technical_features = TechnicalAnalyzer.add_technical_features(stock_data).dropna()
        for index in range(technical_features.shape[0] - DataPreparation.STATE_SIZE):
            states.append(np.array(technical_features[index:index + DataPreparation.STATE_SIZE], dtype=float))

        return states

    @staticmethod
    def calculate_reward_for_state(state, action, next_state) -> float:
        """
        For a given state, action and next state, calculate the reward.
        """
        r_t = next_state[-1, DataPreparation.NORMALIZED_PRICE_INDEX] - state[-1, DataPreparation.NORMALIZED_PRICE_INDEX]
        sigma_target = next_state[:, DataPreparation.NORMALIZED_PRICE_INDEX].var()
        sigma_t_minus_1 = state[:, DataPreparation.NORMALIZED_PRICE_INDEX][:-1].var()
        result = DataPreparation.MU * (action * r_t * sigma_target / sigma_t_minus_1 - DataPreparation.BP *
                                       state[-1, DataPreparation.NORMALIZED_PRICE_INDEX])
        return result
