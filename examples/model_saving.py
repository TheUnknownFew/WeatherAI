from tensorflow.keras import Sequential, layers
import numpy as np
import json


def write_model():
    """
    Simple check to see how tensorflow outputs its models.
    """
    model = Sequential()
    model.add(layers.LSTM(50, activation='relu', input_shape=(3, 1)))
    model.add(layers.Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'cosine_similarity', 'accuracy'])
    with open('model.json', 'w') as f:
        json.dump(json.loads(model.to_json()), f, indent=4)
    print('json has finished!')


if __name__ == '__main__':
    write_model()
