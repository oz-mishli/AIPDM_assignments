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
