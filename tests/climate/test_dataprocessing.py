from AIForecast.modeling.dataprocessing import SupervisedTimeseriesTransformer, RollingSplit, ExpandingSplit, \
    StraightSplit, ZStandardizer
from pandas.util import testing as pdtest
import numpy as np
import pandas as pd
import unittest


class TestTimeseriesSplitter(unittest.TestCase):
    def setUp(self) -> None:
        data_20x = np.array([
            [1, 21],
            [2, 22],
            [3, 23],
            [4, 24],
            [5, 25],
            [6, 26],
            [7, 27],
            [8, 28],
            [9, 29],
            [10, 30],
            [11, 31],
            [12, 32],
            [13, 33],
            [14, 34],
            [15, 35],
            [16, 36],
            [17, 37],
            [18, 38],
            [19, 39],
            [20, 40]
        ])
        data_13x = np.array([
            [1, 21],
            [2, 22],
            [3, 23],
            [4, 24],
            [5, 25],
            [6, 26],
            [7, 27],
            [8, 28],
            [9, 29],
            [10, 30],
            [11, 31],
            [12, 32],
            [13, 33]
        ])
        data_10x = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        cols = [f'col{i}' for i in range(2)]
        self.df_20x = pd.DataFrame(data_20x, columns=cols)
        self.df_13x = pd.DataFrame(data_13x, columns=cols)
        self.df_10x = pd.DataFrame(data_10x, columns=['col0'])

    def tearDown(self) -> None:
        del self.df_20x
        del self.df_13x
        del self.df_10x

    def test_timeseries_transformer(self):
        transformer = SupervisedTimeseriesTransformer(
            input_columns=['col0'],
            output_columns=['col0'],
            input_width=3,
            output_width=3,
            stride=1,
            label_offset=2
        )
        print(transformer(StraightSplit()(self.df_20x)))

    def test_straight_split(self):
        split = StraightSplit()
        print(split(self.df_20x))

    def test_zstandarizer(self):
        split = RollingSplit(3, 3, stride=2)
        print(ZStandardizer(split(self.df_20x))())

    def test_rolling_split(self):
        split_tests = [
            {   # Test 1: Single unit parameters.
                'split': RollingSplit, 'data': self.df_20x,
                'args': {'training_size': 1, 'testing_size': 1, 'validation_size': 1, 'stride': 1, 'gap': 1},
                'expected': {
                    # Ensure the length of the splits are consistently the same across all splits.
                    # Total Split Calculation:
                    # total_splits = ((n_samples - (train_size + test_size + validation_size + gap)) / stride) + 1
                    'lengths': {'total_splits': 17, 'train_len': 1, 'test_len': 1, 'val_len': 1},
                    # Test to see if the window is aligned:
                    # - If the training split is aligned properly, then the other splits are aligned.
                    # training_end = training_size + ((total_splits - 1) * stride)
                    'training_end_idx': 17
                }
            },
            {   # Test 2: No Gap and a larger window length.
                'split': RollingSplit, 'data': self.df_20x,
                'args': {'training_size': 5, 'testing_size': 1, 'validation_size': 2, 'stride': 1},
                'expected': {
                    # Ensure the length of the splits are consistently the same across all splits.
                    # Total Split Calculation:
                    # total_splits = ((n_samples - (train_size + test_size + validation_size + gap)) / stride) + 1
                    'lengths': {'total_splits': 13, 'train_len': 5, 'test_len': 1, 'val_len': 2},
                    # Test to see if the window is aligned:
                    # - If the training split is aligned properly, then the other splits are aligned.
                    # training_end = training_size + ((total_splits - 1) * stride)
                    'training_end_idx': 17
                }
            },
            {   # Test 3: Change in stride
                'split': RollingSplit, 'data': self.df_20x,
                'args': {'training_size': 5, 'testing_size': 1, 'validation_size': 2, 'stride': 3},
                'expected': {
                    # Ensure the length of the splits are consistently the same across all splits.
                    # Total Split Calculation:
                    # total_splits = ((n_samples - (train_size + test_size + validation_size + gap)) / stride) + 1
                    'lengths': {'total_splits': 5, 'train_len': 5, 'test_len': 1, 'val_len': 2},
                    # Test to see if the window is aligned:
                    # - If the training split is aligned properly, then the other splits are aligned.
                    # training_end = training_size + ((total_splits - 1) * stride)
                    'training_end_idx': 17
                }
            },
            {   # Test 4: No Validation Set
                'split': RollingSplit, 'data': self.df_20x,
                'args': {'training_size': 8, 'testing_size': 3},
                'expected': {
                    # Ensure the length of the splits are consistently the same across all splits.
                    # Total Split Calculation:
                    # total_splits = ((n_samples - (train_size + test_size + validation_size + gap)) / stride) + 1
                    'lengths': {'total_splits': 10, 'train_len': 8, 'test_len': 3, 'val_len': 0},
                    # Test to see if the window is aligned:
                    # - If the training split is aligned properly, then the other splits are aligned.
                    # training_end = training_size + ((total_splits - 1) * stride)
                    'training_end_idx': 17
                }
            },
            {   # Test 5: Large Stride and Large Window
                'split': RollingSplit, 'data': self.df_20x,
                'args': {'training_size': 8, 'testing_size': 2, 'stride': 8},
                'expected': {
                    # Ensure the length of the splits are consistently the same across all splits.
                    # Total Split Calculation:
                    # total_splits = ((n_samples - (train_size + test_size + validation_size + gap)) / stride) + 1
                    'lengths': {'total_splits': 2, 'train_len': 8, 'test_len': 2, 'val_len': 0},
                    # Test to see if the window is aligned:
                    # - If the training split is aligned properly, then the other splits are aligned.
                    # training_end = training_size + ((total_splits - 1) * stride)
                    'training_end_idx': 16
                }
            },
            {   # Test 6: Gap & Stride
                'split': RollingSplit, 'data': self.df_20x,
                'args': {'training_size': 5, 'testing_size': 1, 'validation_size': 2, 'gap': 1, 'stride': 5},
                'expected': {
                    # Ensure the length of the splits are consistently the same across all splits.
                    # Total Split Calculation:
                    # total_splits = ((n_samples - (train_size + test_size + validation_size + gap)) / stride) + 1
                    'lengths': {'total_splits': 3, 'train_len': 5, 'test_len': 1, 'val_len': 2},
                    # Test to see if the window is aligned:
                    # - If the training split is aligned properly, then the other splits are aligned.
                    # training_end = training_size + ((total_splits - 1) * stride)
                    'training_end_idx': 15
                }
            }
        ]

        for i, test in enumerate(split_tests):
            print(f'Test {i}:\nargs - {test["args"]}\n')
            splits = test['split'](**test['args'])(test['data'])
            lengths = test['expected']['lengths']
            training_end_idx = test['expected']['training_end_idx']
            training_start_idx = training_end_idx - lengths['train_len']
            for j, split in enumerate(splits):
                print(f'Split {j}:')
                print(f'Train\n{split.train_split}')
                print(f'Val\n{split.validation_split}')
                print(f'Test\n{split.test_split}')
                print('-----------------------\n\n')
                # Test the lengths of the splits to ensure that they are the correct sizes.
                self.assertEqual(len(split.train_split), lengths['train_len'], )
                self.assertEqual(len(split.validation_split), lengths['val_len'])
                self.assertEqual(len(split.test_split), lengths['test_len'])
            # Test to make sure that the number of splits matches the calculated total number of splits.
            self.assertEqual(len(splits), lengths['total_splits'])
            # Assert that the window is still aligned by checking the last window equals the expected range.
            pdtest.assert_frame_equal(test['data'].iloc[training_start_idx:training_end_idx], splits[-1].train_split)

    def test_expanding_split(self):
        split_tests = [
            {
                'split': ExpandingSplit, 'data': self.df_20x,
                'args': {'training_size': 1, 'testing_size': 1, 'validation_size': 1, 'expansion_rate': 1, 'gap': 1},
                'expected': {
                    # Ensure the length of the splits are consistently the same across all splits.
                    # Total Split Calculation:
                    # total_splits =
                    # ((n_samples - (train_size + test_size + validation_size + gap)) / expansion_rate) + 1
                    'total_splits': 17,
                    # Test to see if the window is aligned:
                    # - If the training split is aligned properly, then the other splits are aligned.
                    # training_end = training_size + ((total_splits - 1) * expansion_rate)
                    'training_end_idx': 17
                }
            },
            {
                'split': ExpandingSplit, 'data': self.df_20x,
                'args': {'training_size': 5, 'testing_size': 3, 'validation_size': 3, 'expansion_rate': 3},
                'expected': {
                    # Ensure the length of the splits are consistently the same across all splits.
                    # Total Split Calculation:
                    # total_splits =
                    # ((n_samples - (train_size + test_size + validation_size + gap)) / expansion_rate) + 1
                    'total_splits': 4,
                    # Test to see if the window is aligned:
                    # - If the training split is aligned properly, then the other splits are aligned.
                    # training_end = training_size + ((total_splits - 1) * expansion_rate)
                    'training_end_idx': 14
                }
            },
            {
                'split': ExpandingSplit, 'data': self.df_20x,
                'args': {'training_size': 8, 'testing_size': 2, 'expansion_rate': 5},
                'expected': {
                    # Ensure the length of the splits are consistently the same across all splits.
                    # Total Split Calculation:
                    # total_splits =
                    # ((n_samples - (train_size + test_size + validation_size + gap)) / expansion_rate) + 1
                    'total_splits': 3,
                    # Test to see if the window is aligned:
                    # - If the training split is aligned properly, then the other splits are aligned.
                    # training_end = training_size + ((total_splits - 1) * expansion_rate)
                    'training_end_idx': 18
                }
            },
            {
                'split': ExpandingSplit, 'data': self.df_20x,
                'args': {'training_size': 2, 'testing_size': 2, 'gap': 2},
                'expected': {
                    # Ensure the length of the splits are consistently the same across all splits.
                    # Total Split Calculation:
                    # total_splits =
                    # ((n_samples - (train_size + test_size + validation_size + gap)) / expansion_rate) + 1
                    'total_splits': 15,
                    # Test to see if the window is aligned:
                    # - If the training split is aligned properly, then the other splits are aligned.
                    # training_end = training_size + ((total_splits - 1) * expansion_rate)
                    'training_end_idx': 16
                }
            }
        ]

        for i, test in enumerate(split_tests):
            print(f'Test {i}:\nargs - {test["args"]}\n')
            splits = test['split'](**test['args'])(test['data'])
            total_splits = test['expected']['total_splits']
            training_end_idx = test['expected']['training_end_idx']
            for j, split in enumerate(splits):
                print(f'Split {j}:')
                print(f'Train\n{split.train_split}')
                print(f'Val\n{split.validation_split}')
                print(f'Test\n{split.test_split}')
                print('-----------------------\n\n')
            self.assertEqual(total_splits, len(splits))
            # Assert that the window is still aligned by checking the last window equals the expected range.
            pdtest.assert_frame_equal(test['data'].iloc[:training_end_idx], splits[-1].train_split)


if __name__ == '__main__':
    unittest.main()
