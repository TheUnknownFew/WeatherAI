from typing import List, Callable, Tuple, Union

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer

from AIForecast.sysutils.pathing import FolderStructure
from AIForecast.sysutils.sysexceptions import TimeseriesTransformationError

# ----------------- Data Classes : ----------------- #
#
#
#


class DataSplit:
    """
    A simple container class for a single train, test, and validation split.
    """
    def __init__(self, parent_data: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, validation: pd.DataFrame):
        self.parent_data: pd.DataFrame = parent_data
        self.train_split: pd.DataFrame = train
        self.test_split: pd.DataFrame = test
        self.validation_split: pd.DataFrame = validation
        self.has_validation_set: bool = len(self.validation_split) > 0

    def __repr__(self):
        return f'Training Set:\n' \
               f'{self.train_split}\n' \
               f'Vaidation Split:\n' \
               f'{self.validation_split}\n' \
               f'Test Split:\n' \
               f'{self.test_split}'


class SampleSet:
    def __init__(self):
        self.index = None
        self.__inputs: np.array = np.array([])
        self.__labels: np.array = np.array([])
        self.__append = self.__init_append

    def __init_append(self, sample: pd.DataFrame, labels: pd.DataFrame):
        """
        Ran the first time `append_samples` is called. Initializes the sample sets to the correct
        dimensions for the sample set.
        :param sample:
        :param labels:
        :return:
        """
        self.index = sample.index
        multi_featured = len(labels) > 1 or len(labels.columns) > 1
        self.__inputs = np.array([sample.to_numpy()])
        label_set = np.array([labels.to_numpy()]) if multi_featured else labels.to_numpy().flatten()
        self.__labels = label_set
        self.__append = self.__append_sample

    def __append_sample(self, sample: pd.DataFrame, labels: pd.DataFrame):
        """

        :param sample:
        :param labels:
        :return:
        """
        self.index = self.index.append(sample.index)
        multi_featured = len(labels) > 1 or len(labels.columns) > 1
        self.__inputs = np.concatenate((self.__inputs, [sample.to_numpy()]))
        label_set = np.array([labels.to_numpy()]) if multi_featured else labels.to_numpy().flatten()
        self.__labels = np.concatenate((self.__labels, label_set))

    @property
    def append_sample(self) -> Callable:
        """

        :return:
        """
        return self.__append

    @property
    def samples(self) -> np.array:
        return self.__inputs

    @property
    def labels(self) -> np.array:
        return self.__labels

    def __repr__(self):
        sample_set = f'Input Shape: {self.__inputs.shape}\tOutput Shape: {self.__labels.shape}\n'
        for x, y in zip(self.samples, self.labels):
            sample_set += f'{x} -> {y}\n'
        return sample_set


class TimeseriesData:
    def __init__(self, output_cols: List[str], num_steps: int):
        self.out_cols: List[str] = output_cols
        self.num_steps: int = num_steps
        self.training_samples: SampleSet = SampleSet()
        self.validation_samples: SampleSet = SampleSet()
        self.test_samples: SampleSet = SampleSet()

    def __repr__(self):
        return f'Training Samples:\n' \
               f'{self.training_samples}\n\n' \
               f'Validation Samples:\n' \
               f'{self.validation_samples}\n\n' \
               f'Testing Samples:\n{self.test_samples}'


# ----------- Data Processing Classes : ------------ #
#
#
#


class DataImputer:
    def __init__(self, imputer: str):
        if imputer not in {'Iterative', 'Simple', 'None'}:
            raise ValueError(f'Imputer type "{imputer}" was not recognized as an imputer.')
        self. imputer_type = imputer
        if self.imputer_type == 'Iterative':
            self.imputer = IterativeImputer(missing_values=np.nan,
                                            initial_strategy='most_frequent',
                                            imputation_order='arabic')
        elif self.imputer_type == 'Simple':
            self.imputer = SimpleImputer(missing_values=np.nan)
        elif self.imputer_type == 'None':
            self.imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)

    def __call__(self, data: pd.DataFrame):
        return pd.DataFrame(self.imputer.fit_transform(data), columns=data.columns)


# --------- Pipeline Processing Classes : ---------- #
#
#
#

class StraightSplit:
    def __init__(self, train_split: float = 0.8, validate_split: float = 0.0):
        """
        Creates a simple sequential split consisting of a training split, a validation split, and a testing split.
        Each split consists of a sequential amount of data points from the data set. The training set will contain the
        first data points in the dataset, the validation set will contain the middle data points of the dataset, and
        the testing set will contain every value after the validation set.
        :param train_split: A percent represented by a value between 1 and 0. Defines the percentage of overall data
                the training split set will consist of.
        |       Default value is 0.8, or 80% of the data points are in the training split.
        :param validate_split: A percent represented by a value between 1 and 0. Defines the percentage of overall data
                the validation split set will consist of.
        |       Default value is 0.0, or 0% of the data points are in the validation split. If the split is 0%, then an
                empty set will be returned for the validation set. This 0% value exists if you just want a straight
                training and testing split with no validation split.
        :return: Returns a three sets, the first set being the training set, the second set being the validation set,
                 and the last set being the testing set.
        :raises: Raises a ValueError if the overall split adds up to greater than 1 or is less than 0.
        """
        self.val_split = train_split + validate_split
        if train_split < 0 or validate_split < 0 and self.val_split == 0 and self.val_split > 1:
            raise ValueError(f'The split percentage should be a value between 0 and 1. '
                             f'[0 < train:{train_split} + validate:{validate_split} <= 1]')
        self.train_split = train_split
        self.val_split = validate_split

    def __call__(self, data: pd.DataFrame) -> List[DataSplit]:
        data_len = len(data)
        train_at = int(data_len * self.train_split)
        val_at = int(data_len * self.val_split) + train_at
        train_set = data.iloc[:train_at]
        validate_set = data.iloc[train_at:val_at]
        test_set = data.iloc[val_at:]
        return [DataSplit(data, train_set, test_set, validate_set)]


class RollingSplit:
    def __init__(self, training_size: int, testing_size: int, validation_size: int = 0, stride: int = 1, gap: int = 0):
        """
        Todo: Comment
        :param training_size:
        :param testing_size:
        :param validation_size:
        :param stride:
        :param gap: Do not use with SupervisedTimeseriesTransformer. SupervisedTimeseriesTransformer currently does not
                    not support data that is not fully continuous. Setting a gap will give inconsistent results if used
                    with SupervisedTimeseriesTransformer.
        :return:
        """
        self.win_width = training_size + testing_size + validation_size + gap
        self.training_size = training_size
        self.validation_size = validation_size
        self.testing_size = testing_size
        self.stride = stride
        self.gap = gap

    def __call__(self, data: pd.DataFrame) -> List[DataSplit]:
        sample_size = len(data)
        if self.win_width > sample_size:
            raise IndexError(f'Total window length {self.win_width} exceeds the total number of samples {sample_size}.')
        splits = []
        for idx in [idx for idx in range(0, sample_size, self.stride) if idx + self.win_width <= sample_size]:
            train_end = self.training_size + idx
            val_end = train_end + self.validation_size
            test_end = val_end + self.testing_size + self.gap
            training_split = data.iloc[idx:train_end]
            validation_split = data.iloc[train_end:val_end]
            testing_split = data.iloc[val_end+self.gap:test_end]
            splits.append(DataSplit(data, training_split, testing_split, validation_split))
        return splits


class ExpandingSplit:
    def __init__(self,
                 training_size: int,
                 testing_size: int,
                 validation_size: int = 0,
                 expansion_rate: int = 1,
                 gap: int = 0):
        """
        Todo: Comment
        :param training_size:
        :param testing_size:
        :param validation_size:
        :param expansion_rate:
        :param gap: Do not use with SupervisedTimeseriesTransformer. SupervisedTimeseriesTransformer currently does not
                    not support data that is not fully continuous. Setting a gap will give inconsistent results if used
                    with SupervisedTimeseriesTransformer.
        :return:
        """
        self.win_width = training_size + testing_size + validation_size + gap
        self.training_size = training_size
        self.validation_size = validation_size
        self.testing_size = testing_size
        self.expansion_rate = expansion_rate
        self.gap = gap

    def __call__(self, data: pd.DataFrame) -> List[DataSplit]:
        sample_size = len(data)
        if self.win_width >= sample_size:
            raise IndexError(f'Total window length {self.win_width} exceeds the total number of samples.')
        splits = []
        tail_width = self.win_width - self.training_size
        end_idx = [idx for idx
                   in range(self.training_size, sample_size, self.expansion_rate)
                   if idx + tail_width <= sample_size]
        for train_end in end_idx:
            val_end = train_end + self.validation_size
            test_end = val_end + self.testing_size
            training_split = data.iloc[:train_end]
            validation_split = data.iloc[train_end:val_end]
            testing_split = data.iloc[val_end + self.gap:test_end + self.gap]
            splits.append(DataSplit(data, training_split, testing_split, validation_split))
        return splits


class ZStandardizer:
    def __init__(self, splits: List[DataSplit]):
        self.splits: List[DataSplit] = splits
        self.training_means: List[pd.Series] = [split.train_split.mean() for split in splits]
        self.training_std: List[pd.Series] = [split.train_split.std() for split in splits]

    def __call__(self) -> List[DataSplit]:
        for split, mean, std in zip(self.splits, self.training_means, self.training_std):
            split.train_split = (split.train_split - mean) / std
            split.validation_split = (split.validation_split - mean) / std
            split.test_split = (split.test_split - mean) / std
        return self.splits


class MinMaxNormalizer:
    def __init__(self, splits: List[DataSplit], scale_range: Tuple[int, int] = (0, 1)):
        self.splits: List[DataSplit] = splits
        self.a, self.b = scale_range[0], scale_range[1]
        self.training_min = [split.train_split.min() for split in splits]
        self.training_max = [split.train_split.max() for split in splits]

    def __call__(self) -> List[DataSplit]:
        ab_diff = self.b - self.a
        for split, split_min, split_max in zip(self.splits, self.training_min, self.training_max):
            min_max_diff = split_max - split_min
            split.train_split = self.a + ((split.train_split - split_min) * ab_diff) / min_max_diff
            split.validation_split = self.a + ((split.validation_split - split_min) * ab_diff) / min_max_diff
            split.test_split = self.a + ((split.test_split - split_min) * ab_diff) / min_max_diff
        return self.splits


class SupervisedTimeseriesTransformer:
    def __init__(self,
                 input_columns: List[str],
                 output_columns: List[str],
                 input_width: int = 1,
                 output_width: int = 1,
                 stride: int = 1,
                 label_offset: int = 1):
        self.width_in: int = input_width
        self.width_out: int = output_width
        self.input_columns: List[str] = input_columns
        self.output_columns: List[str] = output_columns
        self.stride: int = stride
        self.label_offset: int = label_offset
        self.window_width: int = input_width + (output_width + label_offset - input_width)

    def __call__(self, splits: List[DataSplit]) -> List[TimeseriesData]:
        return [self.__make_timeseries_samples(split) for split in splits]

    def __make_timeseries_samples(self, split: DataSplit) -> TimeseriesData:
        series = TimeseriesData(self.output_columns, self.width_out)
        self.__make_samples(split.parent_data, split.train_split, series.training_samples)
        self.__make_samples(split.parent_data, split.validation_split, series.validation_samples)
        self.__make_samples(split.parent_data, split.test_split, series.test_samples, True)
        return series

    def __make_samples(self,
                       data_set: pd.DataFrame,
                       split: pd.DataFrame,
                       sample_set: SampleSet,
                       is_test_set: bool = False):
        split_size = len(split)
        for idx in [idx for idx in range(0, split_size, self.stride) if idx + self.width_in <= split_size]:
            end_idx = idx + self.width_out + self.label_offset
            if is_test_set and end_idx - split_size >= 0:
                break
            sample = split[self.input_columns].iloc[idx:self.width_in+idx]
            labels = data_set[self.output_columns].iloc[end_idx-self.width_out:end_idx]
            if len(labels) < self.width_out:
                raise TimeseriesTransformationError(
                    f'input width {self.width_in}, output width {self.width_out}, stride {self.stride},'
                    f' and label offset {self.label_offset} overflowed outside of the bounds of the dataset :'
                    f' length {len(data_set)}')
            sample_set.append_sample(sample, labels)


class ForecastModelTrainer:
    def __init__(self, path_to_model: str):
        with open(path_to_model, 'r') as f:
            self.model: tf.keras.Model = tf.keras.models.model_from_json(f.read())

    def __call__(self,
                 sample_set: List[TimeseriesData],
                 epochs=10,
                 learning_rate=0.001,
                 callbacks: List[tf.keras.callbacks.Callback] = None) -> tf.keras.Model:
        num_features = len(sample_set[0].out_cols)
        steps_out = sample_set[0].num_steps
        output_layer = tf.keras.layers.Dense(units=num_features * steps_out)
        self.model.add(output_layer)
        if steps_out > 1:
            self.model.add(tf.keras.layers.Reshape([steps_out, num_features]))
        self.model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                           loss=['mae', 'mse'],
                           metrics=['mae', 'accuracy', 'cosine_similarity'])
        for samples in sample_set:
            val_set = (samples.validation_samples.samples, samples.validation_samples.labels) \
                if len(samples.validation_samples.samples) > 0 else None
            self.model.fit(
                samples.training_samples.samples,
                samples.training_samples.labels,
                epochs=epochs,
                validation_data=val_set,
                callbacks=callbacks
            )
        return self.model


class ModelEvaluationReporter:
    def __init__(self, trained_model: tf.keras.Model):
        self.model: tf.keras.Model = trained_model
        self.train_fit: pd.DataFrame = None
        self.test_fit: pd.DataFrame = None

    def __call__(self,
                 sample_set: List[TimeseriesData],
                 generate_train_report: bool = True,
                 generate_test_report: bool = True) -> str:
        for sample in sample_set:
            if generate_train_report:
                sample_df = self.__sample_to_dataframe(sample.training_samples, sample.out_cols)
                self.train_fit = sample_df if self.train_fit is None else self.train_fit.append(sample_df)
            if generate_test_report:
                sample_df = self.__sample_to_dataframe(sample.test_samples, sample.out_cols)
                self.test_fit = sample_df if self.test_fit is None else self.test_fit.append(sample_df)
        return 'Yet to be implemented!'

    def __sample_to_dataframe(self, _set: SampleSet, out_cols: List[str]) -> pd.DataFrame:
        cols = out_cols.copy()
        for out in out_cols:
            cols.append(f'{out}_fit')
        ground_truth = _set.labels
        pred = self.model.predict(_set.samples, ground_truth.all())
        model_fit = np.hstack([np.vstack([t for t in ground_truth]), np.vstack([p for p in pred])])
        return pd.DataFrame(model_fit, columns=cols, index=_set.index)

    def save(self, file_loc: str):
        self.train_fit.to_csv(f'{file_loc}_training_report.csv')
        self.test_fit.to_csv(f'{file_loc}_testing_report.csv')
