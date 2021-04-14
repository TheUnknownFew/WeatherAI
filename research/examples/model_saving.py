from sklearn.preprocessing import MinMaxScaler
from tensorflow import TensorShape
from tensorflow.keras import Sequential, layers
import numpy as np
import json
import joblib
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.utils.vis_utils import plot_model
import pandas as pd

from AIForecast.modeling.dataprocessing import SampleSet


def write_model():
    """
    Simple check to see how tensorflow outputs its models.
    """
    my_matrix = np.array([
        [1, 5, 3, 2, 4],
        [4, 10, 30, 6, 2],
        [7, 2, -4, -10, 6],
        [1, 3, 0, 4, 2],
        [100, 3, 300, 2, 0]
    ])

    model = tf.keras.models.Sequential([
        layers.LSTM(32)
    ])
    with open('model.json', 'w') as f:
        json.dump(json.loads(model.to_json()), f, indent=4)
    with open('model.json', 'r') as f:
        model2 = tf.keras.models.model_from_json(f.read())
    model2.add(layers.Dense(1))
    model2.compile('adam', loss='mean_squared_error', metrics=['mae'])
    # # model.save('my_model.h5')
    model2.fit([[[1., 2., 3.]], [[4., 5., 6.]], [[7., 8., 9.]]], [4., 5., 6.])
    model2.summary()
    plot_model(model2, 'test_net.png', show_shapes=True)
    # model = Sequential()
    # model.add(layers.LSTM(50, activation='relu', input_shape=(3, 1)))
    # model.add(layers.Dense(1))
    # model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'cosine_similarity', 'accuracy'])
    # with open('model.json', 'w') as f:
    #     json.dump(json.loads(model.to_json()), f, indent=4)
    # print('json has finished!')


def tmp():
    my_model: Model = load_model('my_model.h5')
    print(isinstance(my_model.layers[1], Normalization))


def rolling(data, window):
    shape = data.shape[:-1] + (data.shape[-1] - window + 1, window)
    strides = data.strides + (data.strides[-1],)
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


class Sample:
    def __init__(self, sample: pd.DataFrame):
        self.sample: pd.DataFrame = sample


def scaler_saving():
    my_matrix = np.array([
        [1, 5, 3, 2, 4],
        [4, 10, 30, 6, 2],
        [7, 2, -4, -10, 6],
        [1, 3, 0, 4, 2],
        [100, 3, 300, 2, 0]
    ])

    df = pd.DataFrame(my_matrix, columns=['col1', 'col2', 'col3', 'col4', 'col5'])
    print(type(df.mean()))
    samples = SampleSet()
    samples.append_sample(df, df)
    samples.append_sample(df, df)
    print(samples.labels)
    # samples = [Sample(df.iloc[i:i+1]) for i in range(3)]
    # df2 = pd.DataFrame(samples, columns=['X'])
    # print(df2.iloc['X'].sample)
    # windows = df.rolling(window=pd.api.indexers.FixedForwardWindowIndexer(window_size=3))
    # for win in windows:
    #     print(win)
    # window = df.rolling(2)
    # for win in window:
    #     print(win)
    # print(my_matrix)
    # m = MinMaxScaler()
    # scaled_matrix = m.fit_transform(my_matrix)
    # print(m.feature_range)


if __name__ == '__main__':
    # tmp()
    # scaler_saving()
    write_model()
    # <tf.Variable 'mean:0' shape=(5,) dtype=float32, numpy=array([22.6,  4.6, 65.8,  0.8,  2.8], dtype=float32)>
