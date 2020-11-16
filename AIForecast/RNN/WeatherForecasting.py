import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import LSTM, Dense
from tensorflow.python.keras.models import Sequential

from AIForecast import utils
from AIForecast.utils import DataUtils, PathUtils


class TimestepBatchGenerator:
    """
    This class is adapted from https://www.tensorflow.org/tutorials/structured_data/time_series.
    Formats the passed data to be used with a neural network as a future predictive model.

    Takes the data passed and splits the data into batches of timed windows.
    """

    def __init__(
            self,
            train_set: pd.DataFrame, validate_set: pd.DataFrame, test_set: pd.DataFrame,
            data_window, label_window, future_window,
            feature_labels=None
    ):
        """
        train_set, validate_set, test_set - the sets used for time windowing.
        data_window - the length of the time window in which training is sampled from. Indicative of X training set.
        label_window - the length of the window indicative of actual results. Corresponds to the length of data used as
        the y set.
        future_window - the length of the time window in which predictions will be made off of. I.e. "Predict x hours
        into the future". Synonymous with time offset.
        feature_labels - a list of labels used for the actual results used to compare to the predicted results.

        The label window is offset from the data window to create a rolling window for predictions.
        """
        self.train_set, self.validate_set, self.test_set = train_set, validate_set, test_set
        self.feature_labels = feature_labels
        if feature_labels is not None:
            self.feature_indices = {name: i for i, name in enumerate(feature_labels)}
        self.column_indices = {name: i for i, name in enumerate(self.train_set.columns)}
        self.data_window, self.label_window, self.offset = data_window, label_window, future_window
        self.window_size = data_window + future_window
        self.slice_data_window = slice(0, data_window)
        self.data_indices = np.arange(self.window_size)[self.slice_data_window]
        self.label_window_begin = self.window_size - self.label_window
        self.slice_label_window = slice(self.label_window_begin, None)
        self.label_indices = np.arange(self.window_size)[self.slice_label_window]

    def split_window(self, features):
        data = features[:, self.slice_data_window, :]
        labels = features[:, self.slice_label_window, :]
        if self.feature_labels is not None:
            labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.feature_labels])
        data.set_shape([None, self.data_window, None])
        labels.set_shape([None, self.label_window, None])
        return data, labels

    def make_dataset(self, data, stride=1, batch_size=32):
        data = np.array(data, dtype=np.float32)
        data_set = timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.window_size,
            sequence_stride=stride,
            shuffle=True,
            batch_size=batch_size
        )
        data_set = data_set.map(self.split_window)
        return data_set

    @property
    def train(self):
        return self.make_dataset(self.train_set)

    @property
    def validate(self):
        return self.make_dataset(self.validate_set)

    @property
    def test(self):
        return self.make_dataset(self.test_set)

    @property
    def example(self):
        result = getattr(self, '_example', None)
        if result is None:
            result = next(iter(self.train))
            self._example = result
        return result


class ForecastingNetwork:

    _MAX_EPOCHS = 50
    """
    The number of times the neural network is fed back.
    """

    def __init__(self, data, use_dropout=True, batch_size=32):
        self.scaler = StandardScaler()
        self.train, self.validate, self.test = DataUtils.split_data(data)
        self.train = self.scaler.fit_transform(self.train)
        self.validate = self.scaler.transform(self.validate)
        self.test = self.scaler.transform(self.test)
        self.model = Sequential([
            LSTM(batch_size, return_sequences=True),
            Dense(units=1)
        ])

    def predict(self, hours_into_the_future, features=None):
        if features is None:
            features = ['temperature']

        batch_generator = TimestepBatchGenerator(
            self.train,
            self.validate,
            self.test,
            hours_into_the_future,
            len(features),
            hours_into_the_future,
            features
        )
        return self._compile_and_fit(batch_generator)

    def _compile_and_fit(self, generator: TimestepBatchGenerator):
        checkpoint = ModelCheckpoint(
            filepath=PathUtils.get_file(PathUtils.get_model_path(), 'model-{epoch:02d}.hdf5'),
            verbose=1
        )
        self.model.compile(
            loss=tf.losses.MeanSquaredError(),
            optimizer=tf.optimizers.Adam(),
            metrics=[tf.metrics.MeanAbsoluteError()]
        )
        utils.log(__name__).debug(self.model.summary())
        return self.model.fit(
            generator.train,
            epochs=ForecastingNetwork._MAX_EPOCHS,
            validation_data=generator.validate,
            callbacks=[checkpoint]
        )
