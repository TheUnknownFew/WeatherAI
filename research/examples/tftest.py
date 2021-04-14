import tensorflow as tf
import numpy as np


class SplitterModule(tf.Module):
    def __init__(self, training_split: float = 0.8):
        super().__init__()
        self.split = tf.Variable(training_split, trainable=False, dtype=tf.float32)

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None))])
    def __call__(self, data_in):
        at = tf.cast(tf.cast(tf.size(data_in), dtype=tf.float32) * self.split, dtype=tf.int32)
        print(tf.identity(tf.size(data_in)))
        return [data_in[:at], data_in[at:]]


if __name__ == '__main__':
    data = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
        [13, 14, 15],
        [16, 17, 18],
        [19, 20, 21],
        [22, 23, 24],
        [25, 26, 27],
        [28, 29, 30]
    ])
    model = SplitterModule()
    print('Values Before Save:')
    out = model(data)
    print(f'{out[0].numpy()}\n{out[1].numpy()}')
    tf.saved_model.save(model, 'ret')
    model2 = tf.saved_model.load('ret')
    print('\n\nValues After Save:')
    out2 = model2(data)
    print(f'{out2[0].numpy()}\n{out2[1].numpy()}')
