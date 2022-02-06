import yfinance as yf
import pandas as pd


class StockDataFetcher:
    @staticmethod
    def fetch_stock_data(symbol: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
        """
        Get stock data by symbol, from start_date to end_date with the interval specified.
        :param symbol: Stock symbol.
        :param start_date: Data from this date on, valid -  (YYYY-MM-DD) str or _datetime..
        :param end_date: Data until this date, valid -  (YYYY-MM-DD) str or _datetime..
        :param interval: Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo.
        """
        return yf.download(symbol, start_date, end_date, interval)
