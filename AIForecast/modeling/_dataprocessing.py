# from enum import Enum
# from typing import List, Callable
#
# from tensorflow.keras import Model, models
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import SimpleImputer, IterativeImputer
# import pandas as pd
# import numpy as np
#
# from AIForecast.sysutils.sysexceptions import TimeseriesTransformationError
#
#
# class DataPipe:
#     def transfer(self):
#         pass
#
#     def receive(self):
#         pass
#
#
# class Pipeline:
#     def __init__(self, pipeline: List[DataPipe]):
#         self.__pipeline: List[DataPipe] = pipeline
#
#     def run(self):
#         for pipe in self.__pipeline:
#             pipe.transfer()
#
#
# class Dataset:
#     def __init__(self, data: pd.DataFrame):
#         self.raw_data: pd.DataFrame = data
#         self.numeric_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
#
#
# class Imputer:
#     pass
#
#
# class DataSplit:
#     """
#     A simple container class for a single train, test, and validation split.
#     """
#
#     def __init__(self, train: pd.DataFrame, test: pd.DataFrame, validation: pd.DataFrame):
#         self.train_split: pd.DataFrame = train
#         self.test_split: pd.DataFrame = test
#         self.validation_split: pd.DataFrame = validation
#         self.has_validation_set: bool = len(self.validation_split) > 0
#
#     @property
#     def data(self) -> pd.DataFrame:
#         return pd.concat((self.train_split, self.validation_split, self.test_split))
#
#
# class SampleSet:
#     def __init__(self):
#         self.__inputs: np.array = np.array([])
#         self.__labels: np.array = np.array([])
#         self.__append = self.__init_append
#
#     def __init_append(self, sample: pd.DataFrame, labels: pd.DataFrame):
#         """
#         Ran the first time `append_samples` is called. Initializes the sample sets to the correct
#         dimensions for the sample set.
#         :param sample:
#         :param labels:
#         :return:
#         """
#         self.__inputs = np.array([sample.to_numpy()])
#         label_set = [labels.to_numpy()] if len(labels) > 1 or len(labels.columns) > 1 else labels.to_numpy().flatten()
#         self.__labels = label_set
#         self.__append = self.__append_sample
#
#     def __append_sample(self, sample: pd.DataFrame, labels: pd.DataFrame):
#         """
#
#         :param sample:
#         :param labels:
#         :return:
#         """
#         self.__inputs = np.concatenate((self.__inputs, [sample.to_numpy()]))
#         label_set = [labels.to_numpy()] if len(labels) > 1 or len(labels.columns) > 1 else labels.to_numpy().flatten()
#         self.__labels = np.concatenate((self.__labels, label_set))
#
#     @property
#     def append_sample(self) -> Callable:
#         """
#
#         :return:
#         """
#         return self.__append
#
#     @property
#     def samples(self) -> np.array:
#         return self.__inputs
#
#     @property
#     def labels(self) -> np.array:
#         return self.__labels
#
#     def __repr__(self):
#         sample_set = f'Input Shape: {self.__inputs.shape}\tOutput Shape: {self.__labels.shape}\n'
#         for x, y in zip(self.samples, self.labels):
#             sample_set += f'{x} -> {y}\n'
#         return sample_set
#
#
# class TimeseriesData:
#     def __init__(self, output_cols: List[str]):
#         self.out_cols: List[str] = output_cols
#         self.training_samples: SampleSet = SampleSet()
#         self.validation_samples: SampleSet = SampleSet()
#         self.test_samples: SampleSet = SampleSet()
#
#
# class SupervisedTimeseriesTransformer:
#     # https://www.tensorflow.org/tutorials/structured_data/time_series
#     def __init__(self,
#                  split: DataSplit,
#                  input_columns: List[str],
#                  output_columns: List[str],
#                  input_width: int = 1,
#                  output_width: int = 1,
#                  stride: int = 1,
#                  label_offset: int = 1):
#         self.split: DataSplit = split
#         self.width_in: int = input_width
#         self.input_columns: List[str] = input_columns
#         self.width_out: int = output_width
#         self.output_columns: List[str] = output_columns
#         self.stride: int = stride
#         self.label_offset: int = label_offset
#         self.window_width = input_width + (output_width + label_offset - input_width)
#         self.__timeseries_data: TimeseriesData = TimeseriesData(output_columns)
#         self.__make_timeseries_samples()
#
#     def __make_timeseries_samples(self):
#         self.__make_samples(self.split.train_split, self.__timeseries_data.training_samples)
#         self.__make_samples(self.split.validation_split, self.__timeseries_data.validation_samples)
#         self.__make_samples(self.split.test_split, self.__timeseries_data.test_samples, True)
#
#     def __make_samples(self, split: pd.DataFrame, sample_set: SampleSet, is_test_set: bool = False):
#         split_size = len(split)
#         data_set = self.split.data
#         for idx in [idx for idx in range(0, split_size, self.stride) if idx + self.width_in <= split_size]:
#             end_idx = idx + self.width_out + self.label_offset
#             if is_test_set and end_idx - split_size >= 0:
#                 break
#             sample = split[self.input_columns].iloc[idx:self.width_in+idx]
#             labels = data_set[self.output_columns].iloc[end_idx-self.width_out:end_idx]
#             if len(labels) < self.width_out:
#                 raise TimeseriesTransformationError(
#                     f'input width {self.width_in}, output width {self.width_out}, stride {self.stride},'
#                     f' and label offset {self.label_offset} overflowed outside of the bounds of the dataset :'
#                     f' length {len(data_set)}')
#             sample_set.append_sample(sample, labels)
#
#     @property
#     def sample_set(self) -> TimeseriesData:
#         return self.__timeseries_data
#
#
# class TimeseriesSplitter:
#     def __init__(self, source: Dataset):
#         self.data = source
#
#     def simple_split(self, train_split: float = 0.8, validate_split: float = 0.0) -> List[DataSplit]:
#         """
#         Creates a simple sequential split consisting of a training split, a validation split, and a testing split.
#         Each split consists of a sequential amount of data points from the data set. The training set will contain the
#         first data points in the dataset, the validation set will contain the middle data points of the dataset, and
#         the testing set will contain every value after the validation set.
#         :param train_split: A percent represented by a value between 1 and 0. Defines the percentage of overall data
#                 the training split set will consist of.
#         |       Default value is 0.8, or 80% of the data points are in the training split.
#         :param validate_split: A percent represented by a value between 1 and 0. Defines the percentage of overall data
#                 the validation split set will consist of.
#         |       Default value is 0.0, or 0% of the data points are in the validation split. If the split is 0%, then an
#                 empty set will be returned for the validation set. This 0% value exists if you just want a straight
#                 training and testing split with no validation split.
#         :return: Returns a three sets, the first set being the training set, the second set being the validation set,
#                  and the last set being the testing set.
#         :raises: Raises a ValueError if the overall split adds up to greater than 1 or is less than 0.
#         """
#         val_split = train_split + validate_split
#         if train_split < 0 or validate_split < 0 and val_split == 0 and val_split > 1:
#             raise ValueError(f'The split percentage should be a value between 0 and 1. '
#                              f'[0 < train:{train_split} + validate:{validate_split} <= 1]')
#         _data = self.data.raw_data
#         train_at = int(len(_data) * train_split)
#         val_at = int(len(_data) * val_split)
#         train_set = _data.iloc[:train_at]
#         validate_set = _data.iloc[train_at:val_at]
#         test_set = _data.iloc[val_at:]
#         return [DataSplit(train_set, validate_set, test_set)]
#
#     def rolling_split(self,
#                       training_size: int,
#                       testing_size: int,
#                       validation_size: int = 0,
#                       stride: int = 1,
#                       gap: int = 0) -> List[DataSplit]:
#         """
#         Todo: Comment
#         :param training_size:
#         :param testing_size:
#         :param validation_size:
#         :param stride:
#         :param gap: Do not use with SupervisedTimeseriesTransformer. SupervisedTimeseriesTransformer currently does not
#                     not support data that is not fully continuous. Setting a gap will give inconsistent results if used
#                     with SupervisedTimeseriesTransformer.
#         :return:
#         """
#         win_width = training_size + testing_size + validation_size + gap
#         sample_size = len(self.data.raw_data)
#         if win_width > sample_size:
#             raise IndexError(f'Total window length {win_width} exceeds the total number of samples {sample_size}.')
#         splits = []
#         _data = self.data.raw_data
#         for idx in [idx for idx in range(0, sample_size, stride) if idx + win_width <= sample_size]:
#             train_end = training_size + idx
#             val_end = train_end + validation_size
#             test_end = val_end + testing_size + gap
#             training_split = _data.iloc[idx:train_end]
#             validation_split = _data.iloc[train_end:val_end]
#             testing_split = _data.iloc[val_end+gap:test_end]
#             splits.append(DataSplit(training_split, testing_split, validation_split))
#         return splits
#
#     def expanding_split(self,
#                         training_size: int,
#                         testing_size: int,
#                         validation_size: int = 0,
#                         expansion_rate: int = 1,
#                         gap: int = 0) -> List[DataSplit]:
#         """
#         Todo: Comment
#         :param training_size:
#         :param testing_size:
#         :param validation_size:
#         :param expansion_rate:
#         :param gap: Do not use with SupervisedTimeseriesTransformer. SupervisedTimeseriesTransformer currently does not
#                     not support data that is not fully continuous. Setting a gap will give inconsistent results if used
#                     with SupervisedTimeseriesTransformer.
#         :return:
#         """
#         win_width = training_size + testing_size + validation_size + gap
#         sample_size = len(self.data.raw_data)
#         if win_width >= sample_size:
#             raise IndexError(f'Total window length {win_width} exceeds the total number of samples.')
#         splits = []
#         _data = self.data.raw_data
#         tail_width = win_width - training_size
#         end_idx = [idx for idx in range(training_size, sample_size, expansion_rate) if idx + tail_width <= sample_size]
#         for train_end in end_idx:
#             val_end = train_end + validation_size
#             test_end = val_end + testing_size
#             training_split = _data.iloc[:train_end]
#             validation_split = _data.iloc[train_end:val_end]
#             testing_split = _data.iloc[val_end+gap:test_end+gap]
#             splits.append(DataSplit(training_split, testing_split, validation_split))
#         return splits
#
#
# class Normalizer:
#     # Do regular normalization for now, but consider rolling window normalization in the future.
#     def __init__(self, data):
#         self.data: DataSplit = data
#
#     def normalize(self):
#         pass
#
#     def denormalize(self, data: pd.DataFrame, columns: List[str]):
#         pass
#
#
# class ZStandardizer(Normalizer):
#     def __init__(self, data: DataSplit):
#         super().__init__(data)
#         self.train_mean: float = data.train_split.mean()
#         self.train_std: float = data.train_split.std()
#
#     def normalize(self) -> DataSplit:
#         self.data.train_split = (self.data.train_split - self.train_mean) / self.train_std
#         self.data.validation_split = (self.data.validation_split - self.train_mean) / self.train_std
#         self.data.test_split = (self.data.test_split - self.train_mean) / self.train_mean
#         return self.data
#
#
# class MinMaxNormalizer(Normalizer):
#     def __init__(self, data: DataSplit):
#         super().__init__(data)
#         self.train_min = data.train_split.min()
#         self.train_max = data.train_split.max()
#
#     def normalize(self):
#         self.data.train_split = (self.data.train_split - self.train_min) / (self.train_max - self.train_min)
#         self.data.validation_split = (self.data.validation_split - self.train_min) / (self.train_max - self.train_min)
#         self.data.test_split = (self.data.test_split - self.train_min) / (self.train_max - self.train_min)
#         return self.data
#
#
# class DataModeler:
#     # Metric: Mean Absolute Percent Error
#     def __init__(self, path_to_model: str):
#         self.model: Model = models.load_model(path_to_model)
#         self.is_trained: bool = False
#
#     def train_model(self):
#         pass
#
# # File Selector : csv file -> Training
# # File Selector : model schema -> Training
# # Drop down to select an imputer : None, Simple, Iterative
# # Drop down to select a split type : Straight, Rolling, Expanding
#
# # Straight split:
#     # Training percentage: a % -> 80% or 0.8 Optional
#     # Validation percentage: a % -> 0% or 0.0 Optional
#
# # Rolling Split:
#     # training size: an integer or percent -> 0.2 or 20%
#     # validation size: an integer -> 0%
#     # testing size: an integer -> 10%
#     # gap size: an integer -> 0
#     # stride: an integers -> 1
#
# # Expanding Split:
#     # initial training size: an integer or percentage
#     # validation size: an integer or percentage 0%
#     # gap size: an integer -> 0
#     # expansion rate: an integer -> 1
#
# # Drop down to select normalization type: Min-Max norm, Z-standardization
# # Training Features = [], output features = []
# # Input width: integer -> 3
# # Output width: integer -> 1
# # stride: integer -> 1
# # gap: integer -> 0
#
# # Text field for hyper-parameters <- may or may not get implemented
#
# # Button <label train model>
# # Output for a model evaluation and data plots. -> data plots will plot the predictions of the train and test data.
# # Button to save trained model.
#
# # Test menu:
# # File Selector: trained model file.
# # Time horizon: integer -> months
# # Submit button
# # Output window with a plot plotting a graph from the end of the data to the future time horizon, plotting the
# # forecast into the future.
