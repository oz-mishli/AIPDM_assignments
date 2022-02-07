import numpy as np
import pandas as pd


class TechnicalAnalyzer:
    @staticmethod
    def add_technical_features(stock_data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical features to the stock data Dataframe.
        :param stock_data: Stock data we get from StockDataFetcher module.
        """
        pass

    @staticmethod
    def get_normalized_price(stock_data: pd.DataFrame) -> pd.Series:
        """
        Normalize the stock close data to be in range of 0 and 1.
        """
        close_data_price = stock_data['Close']
        min_close_price_value = min(close_data_price)
        close_data_price_range = max(close_data_price) - min_close_price_value
        return (close_data_price - min_close_price_value) / close_data_price_range

    @staticmethod
    def get_return(stock_data: pd.DataFrame, current_index: int, past_index: int, variance_span: int,
                   exp_weight: float) -> float:
        """
        Calculate return relevant to past price.
        """
        if current_index < past_index:
            raise ValueError('Past index should be larger than current_index.')

        close_data = stock_data['Close']
        r_t = close_data[current_index] - close_data[past_index]
        daily_r_t = pd.Series(np.array(close_data[current_index - variance_span:current_index]) - \
                              np.array(close_data[current_index - variance_span - 1:current_index - 1]))
        weights = np.array([max(exp_weight ** power, 0.0001) for power in range(variance_span - 1, -1, -1)]) * (
                1 - exp_weight)
        sigma_t = (daily_r_t * weights).var()
        return r_t / (sigma_t * np.sqrt(current_index - past_index))
