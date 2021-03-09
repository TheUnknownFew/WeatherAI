import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit


class TimeseriesSplitter:

    __type = ['rolling', 'sliding', 'expanding']

    def __init__(self, window_type='sliding'):
        if window_type not in TimeseriesSplitter.__type:
            raise ValueError(f'{window_type} is an invalid split type.')
        self.window_type = window_type

    def split(self):
        pass


class TimeseriesData:
    def __init__(self, data: pd.DataFrame, scaler=MinMaxScaler()):
        """
        In order to use this class, the following conditions must be met:
        - The input pandas dataframe must have a datetime index.
        - The input pandas dataframe must be numerical data.

        :param data: A pandas data frame that contains the Timeseries data.
        :param scaler: An sklearn scaler.
        """
        self.data: pd.DataFrame = data
        self.scaler = scaler
        self.scaled_data: pd.DataFrame = pd.DataFrame(self.scaler.fit_transform(self.data), columns=self.data.columns)
        self.scaled_data.index = self.data.index


class ForecastingModel:
    def __init__(self):
        # Implement
        pass


if __name__ == '__main__':
    # Used for debugging.
    pass
