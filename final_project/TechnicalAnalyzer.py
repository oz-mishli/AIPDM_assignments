import math

import numpy as np
import pandas as pd
from ta import momentum


class TechnicalAnalyzer:
    @staticmethod
    def add_technical_features(stock_data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical features to the stock data Dataframe.
        :param stock_data: Stock data we get from StockDataFetcher module.
        """
        normalized_price = TechnicalAnalyzer.get_normalized_price(stock_data)
        return_indicator_1_month = []
        return_indicator_2_month = []
        return_indicator_3_month = []
        return_indicator_1_year = []
        macd = []
        rsi = []
        for i in range(stock_data.shape[0]):
            return_indicator_1_month.append(TechnicalAnalyzer.get_return(stock_data, i, i - 21, 60, 0.94))
            return_indicator_2_month.append(TechnicalAnalyzer.get_return(stock_data, i, i - 42, 60, 0.94))
            return_indicator_3_month.append(TechnicalAnalyzer.get_return(stock_data, i, i - 63, 60, 0.94))
            return_indicator_1_year.append(TechnicalAnalyzer.get_return(stock_data, i, i - 252, 60, 0.94))
            macd.append(TechnicalAnalyzer.get_macd(stock_data, i, 252, 63, 0.94, 10, 20))
            rsi.append(TechnicalAnalyzer.get_rsi(stock_data, i, 30))

        features = np.concatenate((
            np.array([normalized_price]),
            np.array([return_indicator_1_month]),
            np.array([return_indicator_2_month]),
            np.array([return_indicator_3_month]),
            np.array([return_indicator_1_year]),
            np.array([macd]),
            np.array([rsi])),
            axis=0)

        return pd.DataFrame(features).transpose()

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
            raise ValueError('current_index should be larger than Past index.')

        if past_index < 0 or current_index < variance_span + 1:
            return None

        close_data = stock_data['Close']
        r_t = close_data[current_index] - close_data[past_index]
        daily_r_t = pd.Series(np.array(close_data[current_index - variance_span:current_index]) -
                              np.array(close_data[current_index - variance_span - 1:current_index - 1]))
        sigma_t = (TechnicalAnalyzer._get_exponential_moving_average(daily_r_t, exp_weight)).var()
        return r_t / max((sigma_t * np.sqrt(current_index - past_index)), 0.01)

    @staticmethod
    def get_macd(stock_data: pd.DataFrame, current_index: int, price_variance_span: int,
                 q_variance_span: int, exp_weight: float, short_range: int, long_range: int) -> float:
        """
        Calculate MACD feature by the article
        """
        current_close_price = stock_data['Close'][:current_index]
        intervals = len(current_close_price)
        q_t = pd.Series(
            [TechnicalAnalyzer._get_q_t(stock_data, index, price_variance_span, exp_weight, short_range, long_range)
             for index in range(intervals - q_variance_span, intervals)])
        q_t_variance = q_t.var()
        result = q_t[len(q_t) - 1] / max(q_t_variance, 0.01)
        if math.isnan(result):
            return None
        return result

    @staticmethod
    def get_rsi(stock_data: pd.DataFrame, current_index: int, window: int) -> float:
        """
        Calculate RSI feature value for current index.
        """
        if current_index < window:
            return None

        rsi = momentum.rsi(stock_data['Close'][:current_index], window, fillna=True)
        return rsi[len(rsi) - 1]

    @staticmethod
    def _get_q_t(stock_data: pd.DataFrame, current_index: int, price_variance_span: int,
                 exp_weight: float, short_range: int, long_range: int) -> float:
        """
        Calculate q_t for MACD indicator.
        """
        current_close_price = stock_data['Close'][:current_index]
        price_variance = TechnicalAnalyzer._get_variance(current_close_price, price_variance_span)
        short_moving_average = TechnicalAnalyzer._get_exponential_moving_average(current_close_price[-1 * short_range:],
                                                                                 exp_weight)
        long_moving_average = TechnicalAnalyzer._get_exponential_moving_average(current_close_price[-1 * long_range:],
                                                                                exp_weight)
        return (short_moving_average - long_moving_average) / price_variance

    @staticmethod
    def _get_exponential_moving_average(data_series: pd.Series, exp_weight: float) -> pd.Series:
        """
        Calculate exponential moving average for given data and lambda parameter.
        """
        weights = np.array([max(exp_weight ** power, 0.0001) for power in range(data_series.shape[0] - 1, -1, -1)]) * (
                1 - exp_weight)
        return (data_series * weights).mean()

    @staticmethod
    def _get_variance(data_series: pd.Series, variance_span: int) -> float:
        """
        Calculate variance of data series over variance_span intervals.
        """
        return data_series[-1 * variance_span:].var()
