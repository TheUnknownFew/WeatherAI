import tkinter as tk
from tkinter import filedialog as fdiag
import os

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
from pandastable import Table
from tensorflow import keras

from AIForecast.modeling import dataprocessing as pipeline
from AIForecast.modeling.dataprocessing import ModelEvaluationReporter
from AIForecast.modeling.tfcallbacks import OutputEpoch
from AIForecast.ui.widgets import MenuWindow, OutputWindow
from AIForecast.ui import constants as ui
from AIForecast.weather import ClimateAccess


class MainMenu(MenuWindow):
    LABEL_TEXT = """
    Training AI to Predict Temperatures
    by
    Alexander Cherry, Anthony Ernst, and Marcus Kline

    Previously worked on by:
    Jason Sutton, Swatt Zhou, and Tiger Goodbread
    """

    def __init__(self, app_frame: tk.Frame):
        super().__init__(app_frame)
        self.label = None

    def init_ui(self):
        super().init_ui()
        self.label = tk.Label(self.body, text=self.LABEL_TEXT, bg=ui.BACKGROUND_COLOR, fg=ui.FOREGROUND_COLOR)

    def draw(self):
        super().draw()
        self.label.place(relx=0, rely=0, relwidth=1, relheight=0.5)

    def hide(self):
        super().hide()
        self.label.destroy()


class TestMenu(MenuWindow):
    def __init__(self, app_frame: tk.Frame):
        super().__init__(app_frame)
        self.upload_trained_model_button = None
        self.trained_model = None
        self.output_frame = None
        self.input_frame = None
        self.time_horizon = None
        self.time_horizon_label = None
        self.test_button = None
        self.output_text_label = None

    def init_ui(self):
        super().init_ui()
        self.input_frame = tk.Frame(master=self.body, bg=ui.BACKGROUND_COLOR)
        self.output_frame = tk.Frame(master=self.body, bg=ui.BACKGROUND_COLOR)
        self.upload_trained_model_button = tk.Button(
            self.input_frame,
            text="Upload Trained Model",
            bg=ui.BUTTON_BACKGROUND,
            fg=ui.BUTTON_FOREGROUND,
            command=lambda: self.upload_trained_model()
        )
        self.time_horizon_label = tk.Label(self.input_frame, text="Time Horizon (Hours)", bg=ui.BACKGROUND_COLOR,
                                           fg=ui.FOREGROUND_COLOR, anchor="w")
        self.time_horizon = tk.Text(self.input_frame)
        self.test_button = tk.Button(
            self.input_frame,
            text="Test",
            bg=ui.BUTTON_BACKGROUND,
            fg=ui.BUTTON_FOREGROUND,
            command=lambda: self.test_model()
        )
        self.output_text_label = tk.Label(self.output_frame, text="Output:", bg=ui.BACKGROUND_COLOR, fg=ui.FOREGROUND_COLOR,
                                          anchor="w")
        self.output_text = OutputWindow(self.output_frame)
        self.output_text.init_ui()

    def draw(self):
        super().draw()
        self.output_frame.place(relx=.2, rely=0, relwidth=.8, relheight=1)
        self.input_frame.place(relx=0, rely=0, relwidth=.2, relheight=1)
        self.upload_trained_model_button.place(x=ui.ALIGN_X + 5, y=ui.ALIGN_Y + 5, width=150)
        self.time_horizon_label.place(x=ui.ALIGN_X + 5, y=ui.ALIGN_Y + 45, width=150)
        self.time_horizon.place(x=ui.ALIGN_X + 5, y=ui.ALIGN_Y + 70, width=150, height=25)
        self.output_text_label.place(relx=.05)
        self.output_text.draw(relx=.05, rely=.05, relwidth=.9, relheight=.9)
        self.test_button.place(x=ui.ALIGN_X + 5, y=ui.ALIGN_Y + 125, width=150)

    def hide(self):
        super().hide()
        self.upload_trained_model_button.destroy()
        self.output_frame.destroy()
        self.input_frame.destroy()
        self.time_horizon.destroy()
        self.time_horizon_label.destroy()
        self.test_button.destroy()
        self.output_text_label.destroy()

    def upload_trained_model(self):
        """
        Author: Marcus Kline
        Purpose: This function is used to upload an already trained model that would be used to test
        :return:
        """
        new_data = fdiag.askopenfile(mode='r', filetypes=[(ui.MODEL_FILE_LABEL, ui.MODEL_FILE_TYPE)])
        if new_data is not None:
            self.trained_model = pd.read_csv(new_data)

    def test_model(self):
        pass


class TrainMenu(MenuWindow):
    def __init__(self, app_frame: tk.Frame):
        super().__init__(app_frame)
        self.normalization_options = ('Min-Max', 'Z Standardization')
        self.normalization_selection = tk.StringVar()
        self.split_type_options = ("Straight Split", "Rolling Split", "Expanding Split")
        self.split_type_selection = tk.StringVar()
        self.imputer_options = ("None", "Simple", "Iterative")
        self.imputer_selection = tk.StringVar()
        self.imputer_selector_label = tk.Label()
        self.csv_selector = None
        self.schema_selector = None
        self.imputer_selector = None
        self.split_type_selector = None
        self.straight_training_slider = None
        self.straight_validation_slider = None
        self.rolling_training_size = None
        self.rolling_validation_size = None
        self.rolling_testing_size = None
        self.rolling_gap_size = None
        self.rolling_stride_size = None
        self.expanding_training_size = None
        self.expanding_validation_size = None
        self.expanding_testing_size = None
        self.expanding_gap_size = None
        self.expanding_expansion_rate = None
        self.normalization_selector = None
        self.training_features = None
        self.output_features = None
        self.input_width = None
        self.output_width = None
        self.stride = None
        self.time_offset = None
        self.epoch = None
        self.train_model_button = None
        self.output_text: OutputWindow = None
        self.save_model_button = None
        self.input_frame = None
        self.output_frame = None
        self.training_csv = None
        self.path_to_model_schema = None
        self.split_type_selector_label = None
        self.normalization_selector_label = None
        self.straight_training_slider_label = None
        self.straight_validation_slider_label = None
        self.rolling_training_size_label = None
        self.rolling_validation_size_label = None
        self.rolling_testing_size_label = None
        self.rolling_gap_size_label = None
        self.rolling_stride_size_label = None
        self.expanding_training_size_label = None
        self.expanding_validation_size_label = None
        self.expanding_testing_size_label = None
        self.expanding_gap_size_label = None
        self.expanding_expansion_rate_label = None
        self.training_features_label = None
        self.output_features_label = None
        self.epoch_label = None
        self.input_width_label = None
        self.output_width_label = None
        self.stride_label = None
        self.time_offset_label = None
        self.output_text_label = None
        self.learning_rate = None
        self.learning_rate_label = None
        self.trained_model = None
        self.csv_selection_label = None
        self.schema_selection_label = None
        self.path_to_csv = None
        self.model_fit_reporter: ModelEvaluationReporter = None

    def init_ui(self):
        super().init_ui()
        self.input_frame = tk.Frame(master=self.body, bg=ui.BACKGROUND_COLOR)
        self.output_frame = tk.Frame(master=self.body, bg=ui.BACKGROUND_COLOR)
        self.csv_selector = tk.Button(
            self.input_frame,
            text="Upload CSV",
            bg=ui.BUTTON_BACKGROUND,
            fg=ui.BUTTON_FOREGROUND,
            command=lambda: self.upload_csv()
        )
        self.csv_selection_label = tk.Label(self.input_frame, text="No File Selected.", bg=ui.BACKGROUND_COLOR,
                                            fg=ui.FOREGROUND_COLOR, anchor="w"
                                            )
        self.schema_selector = tk.Button(
            self.input_frame,
            text="Upload Model Schema",
            bg=ui.BUTTON_BACKGROUND,
            fg=ui.BUTTON_FOREGROUND,
            command=lambda: self.upload_schema()
        )
        self.schema_selection_label = tk.Label(self.input_frame, text="No File Selected.", bg=ui.BACKGROUND_COLOR,
                                               fg=ui.FOREGROUND_COLOR, anchor="w"
                                               )
        self.imputer_selection.set(self.imputer_options[0])
        self.imputer_selector = tk.OptionMenu(
            self.input_frame,
            self.imputer_selection,
            *self.imputer_options
        )
        self.imputer_selector_label = tk.Label(self.input_frame, text="Imputer Type:", bg=ui.BACKGROUND_COLOR,
                                               fg=ui.FOREGROUND_COLOR, anchor="e")
        self.split_type_selection.set(self.split_type_options[0])
        self.split_type_selector = tk.OptionMenu(
            self.input_frame,
            self.split_type_selection,
            *self.split_type_options,
            command=self.draw_split_type_inputs
        )
        self.split_type_selector_label = tk.Label(self.input_frame, text="Split Type:", bg=ui.BACKGROUND_COLOR,
                                                  fg=ui.FOREGROUND_COLOR, anchor="e")
        self.straight_training_slider = tk.Scale(self.input_frame, from_=0, to=100, tickinterval=1,
                                                 orient=tk.HORIZONTAL)
        self.straight_training_slider.set(80)
        self.straight_training_slider_label = tk.Label(self.input_frame, text="Training Percent:", bg=ui.BACKGROUND_COLOR,
                                                       fg=ui.FOREGROUND_COLOR, anchor="e")
        self.straight_validation_slider = tk.Scale(self.input_frame, from_=0, to=100, tickinterval=1,
                                                   orient=tk.HORIZONTAL)
        self.straight_validation_slider.set(0)
        self.straight_validation_slider_label = tk.Label(self.input_frame, text="Validation Percent:",
                                                         bg=ui.BACKGROUND_COLOR, fg=ui.FOREGROUND_COLOR, anchor="e")
        self.rolling_training_size = tk.Text(self.input_frame)
        self.rolling_training_size.insert(tk.END, '20')
        self.rolling_training_size_label = tk.Label(self.input_frame, text="Training Size:", bg=ui.BACKGROUND_COLOR,
                                                    fg=ui.FOREGROUND_COLOR, anchor="e")
        self.rolling_validation_size = tk.Text(self.input_frame)
        self.rolling_validation_size.insert(tk.END, '0')
        self.rolling_validation_size_label = tk.Label(self.input_frame, text="Validation Size:", bg=ui.BACKGROUND_COLOR,
                                                      fg=ui.FOREGROUND_COLOR, anchor="e")
        self.rolling_testing_size = tk.Text(self.input_frame)
        self.rolling_testing_size.insert(tk.END, '10')
        self.rolling_testing_size_label = tk.Label(self.input_frame, text="Testing Size:", bg=ui.BACKGROUND_COLOR,
                                                   fg=ui.FOREGROUND_COLOR, anchor="e")
        self.rolling_gap_size = tk.Text(self.input_frame)
        self.rolling_gap_size.insert(tk.END, '0')
        self.rolling_gap_size_label = tk.Label(self.input_frame, text="Gap Size:", bg=ui.BACKGROUND_COLOR,
                                               fg=ui.FOREGROUND_COLOR, anchor="e")
        self.rolling_stride_size = tk.Text(self.input_frame)
        self.rolling_stride_size.insert(tk.END, '1')
        self.rolling_stride_size_label = tk.Label(self.input_frame, text="Stride Size:", bg=ui.BACKGROUND_COLOR,
                                                  fg=ui.FOREGROUND_COLOR, anchor="e")
        self.expanding_training_size = tk.Text(self.input_frame)
        self.expanding_training_size.insert(tk.END, '20')
        self.expanding_training_size_label = tk.Label(self.input_frame, text="Training Size:", bg=ui.BACKGROUND_COLOR,
                                                      fg=ui.FOREGROUND_COLOR, anchor="e")
        self.expanding_validation_size = tk.Text(self.input_frame)
        self.expanding_validation_size.insert(tk.END, '0')
        self.expanding_validation_size_label = tk.Label(self.input_frame, text="Validation Size:", bg=ui.BACKGROUND_COLOR,
                                                        fg=ui.FOREGROUND_COLOR, anchor="e")

        self.expanding_testing_size = tk.Text(self.input_frame)
        self.expanding_testing_size.insert(tk.END, '10')
        self.expanding_testing_size_label = tk.Label(self.input_frame, text="Testing Size:", bg=ui.BACKGROUND_COLOR,
                                                     fg=ui.FOREGROUND_COLOR, anchor="e")

        self.expanding_gap_size = tk.Text(self.input_frame)
        self.expanding_gap_size.insert(tk.END, '0')
        self.expanding_gap_size_label = tk.Label(self.input_frame, text="Gap Size:", bg=ui.BACKGROUND_COLOR,
                                                 fg=ui.FOREGROUND_COLOR, anchor="e")
        self.expanding_expansion_rate = tk.Text(self.input_frame)
        self.expanding_expansion_rate.insert(tk.END, '1')
        self.expanding_expansion_rate_label = tk.Label(self.input_frame, text="Expansion Rate:", bg=ui.BACKGROUND_COLOR,
                                                       fg=ui.FOREGROUND_COLOR, anchor="e")
        self.normalization_selection.set(self.normalization_options[0])
        self.normalization_selector = tk.OptionMenu(
            self.input_frame,
            self.normalization_selection,
            *self.normalization_options
        )
        self.normalization_selector_label = tk.Label(self.input_frame, text="Normalization Type:",
                                                     bg=ui.BACKGROUND_COLOR, fg=ui.FOREGROUND_COLOR, anchor="e")
        self.training_features = tk.Listbox(self.input_frame, exportselection=0, selectmode=tk.EXTENDED)
        self.training_features_label = tk.Label(self.input_frame, text="Training Features:", bg=ui.BACKGROUND_COLOR,
                                                fg=ui.FOREGROUND_COLOR, anchor="e")
        self.output_features = tk.Listbox(self.input_frame, exportselection=0, selectmode=tk.EXTENDED)
        self.output_features_label = tk.Label(self.input_frame, text="Output Features:", bg=ui.BACKGROUND_COLOR,
                                              fg=ui.FOREGROUND_COLOR, anchor="e")
        self.input_width = tk.Text(self.input_frame)
        self.input_width.insert(tk.END, '3')
        self.input_width_label = tk.Label(self.input_frame, text="Input Width:", bg=ui.BACKGROUND_COLOR,
                                          fg=ui.FOREGROUND_COLOR, anchor="e")
        self.output_width = tk.Text(self.input_frame)
        self.output_width.insert(tk.END, '1')
        self.output_width_label = tk.Label(self.input_frame, text="Output Width:", bg=ui.BACKGROUND_COLOR,
                                           fg=ui.FOREGROUND_COLOR, anchor="e")
        self.stride = tk.Text(self.input_frame)
        self.stride.insert(tk.END, '1')
        self.stride_label = tk.Label(self.input_frame, text="Stride Size:", bg=ui.BACKGROUND_COLOR,
                                     fg=ui.FOREGROUND_COLOR, anchor="e")
        self.time_offset = tk.Text(self.input_frame)
        self.time_offset.insert(tk.END, '1')
        self.time_offset_label = tk.Label(self.input_frame, text="Time offset:", bg=ui.BACKGROUND_COLOR,
                                          fg=ui.FOREGROUND_COLOR, anchor="e")
        self.epoch = tk.Text(self.input_frame)
        self.epoch.insert(tk.END, '30')
        self.epoch_label = tk.Label(self.input_frame, text="Epoch:", bg=ui.BACKGROUND_COLOR,
                                    fg=ui.FOREGROUND_COLOR, anchor="e")
        self.learning_rate = tk.Text(self.input_frame)
        self.learning_rate.insert(tk.END, '0.0001')
        self.learning_rate_label = tk.Label(self.input_frame, text="Learning Rate:", bg=ui.BACKGROUND_COLOR,
                                            fg=ui.FOREGROUND_COLOR, anchor="e")
        self.train_model_button = tk.Button(
            self.input_frame,
            text="Train Model",
            bg=ui.BUTTON_BACKGROUND,
            fg=ui.BUTTON_FOREGROUND,
            command=lambda: self.train_model()
        )
        self.output_text = OutputWindow(self.output_frame)
        self.output_text.init_ui()
        self.output_text_label = tk.Label(self.output_frame, text="Output:", bg=ui.BACKGROUND_COLOR, fg=ui.FOREGROUND_COLOR)
        self.save_model_button = tk.Button(
            self.input_frame,
            text="Save Model",
            bg=ui.BUTTON_BACKGROUND,
            fg=ui.BUTTON_FOREGROUND,
            command=lambda: self.save_model()
        )

    def draw(self):
        MenuWindow.draw(self)
        self.input_frame.place(relx=0, rely=0, relwidth=1, relheight=.8)
        self.output_frame.place(relx=0, rely=.8, relwidth=1, relheight=.2)
        self.csv_selector.place(x=ui.ALIGN_X + ui.LEFT_BUTTON_ALIGNMENT_OFFSET,
                                y=ui.ALIGN_Y,
                                width=150)
        self.csv_selection_label.place(x=ui.ALIGN_X + ui.X_ELEMENT_OFFSET + ui.X_150_UNIT_WIDTH_OFFSET
                                         + ui.LEFT_BUTTON_ALIGNMENT_OFFSET,
                                       y=ui.ALIGN_Y,
                                       width=200)
        self.schema_selector.place(x=ui.ALIGN_X + ui.RIGHT_BUTTON_ALIGNMENT_OFFSET,
                                   y=ui.ALIGN_Y,
                                   width=150)
        self.schema_selection_label.place(x=ui.ALIGN_X + ui.X_ELEMENT_OFFSET + ui.X_150_UNIT_WIDTH_OFFSET
                                            + ui.RIGHT_BUTTON_ALIGNMENT_OFFSET,
                                          y=ui.ALIGN_Y,
                                          width=200)
        self.imputer_selector_label.place(x=ui.ALIGN_X,
                                          y=ui.ALIGN_Y + ui.Y_ELEMENT_OFFSET,
                                          width=100)
        self.imputer_selector.place(x=ui.ALIGN_X + ui.LABEL_OFFSET + ui.X_100_UNIT_WIDTH_OFFSET,
                                    y=ui.ALIGN_Y + ui.Y_ELEMENT_OFFSET,
                                    width=100)
        self.split_type_selector_label.place(x=ui.ALIGN_X + (ui.LABEL_OFFSET * 2) + ui.X_ELEMENT_OFFSET
                                               + (ui.X_100_UNIT_WIDTH_OFFSET * 2),
                                             y=ui.ALIGN_Y + ui.Y_ELEMENT_OFFSET,
                                             width=100)
        self.split_type_selector.place(x=ui.ALIGN_X + (ui.LABEL_OFFSET * 2) + ui.X_ELEMENT_OFFSET
                                         + (ui.X_100_UNIT_WIDTH_OFFSET * 3),
                                       y=ui.ALIGN_Y + ui.Y_ELEMENT_OFFSET,
                                       width=150)
        self.normalization_selector_label.place(x=ui.ALIGN_X + (ui.LABEL_OFFSET * 2) + (ui.X_ELEMENT_OFFSET * 2)
                                                  + (ui.X_100_UNIT_WIDTH_OFFSET * 3)
                                                  + ui.X_150_UNIT_WIDTH_OFFSET,
                                                y=ui.ALIGN_Y + ui.Y_ELEMENT_OFFSET,
                                                width=110)
        self.normalization_selector.place(x=ui.ALIGN_X + (ui.LABEL_OFFSET * 3) + (ui.X_ELEMENT_OFFSET * 2)
                                            + (ui.X_100_UNIT_WIDTH_OFFSET * 3)
                                            + ui.X_150_UNIT_WIDTH_OFFSET + ui.X_110_UNIT_WIDTH_OFFSET,
                                          y=ui.ALIGN_Y + ui.Y_ELEMENT_OFFSET,
                                          width=150)
        self.training_features_label.place(x=ui.ALIGN_X,
                                           y=ui.ALIGN_Y + (ui.Y_ELEMENT_OFFSET * 4),
                                           width=100)
        self.training_features.place(x=ui.ALIGN_X + ui.LABEL_OFFSET + ui.X_100_UNIT_WIDTH_OFFSET,
                                     y=ui.ALIGN_Y + (ui.Y_ELEMENT_OFFSET * 4),
                                     height=100, width=200)
        self.output_features_label.place(x=ui.ALIGN_X + ui.LABEL_OFFSET + ui.X_ELEMENT_OFFSET
                                           + ui.X_100_UNIT_WIDTH_OFFSET
                                           + ui.X_200_UNIT_WIDTH_OFFSET,
                                         y=ui.ALIGN_Y + (ui.Y_ELEMENT_OFFSET * 4),
                                         width=100)
        self.output_features.place(x=ui.ALIGN_X + (ui.LABEL_OFFSET * 2) + ui.X_ELEMENT_OFFSET
                                     + (ui.X_100_UNIT_WIDTH_OFFSET * 2)
                                     + ui.X_200_UNIT_WIDTH_OFFSET,
                                   y=ui.ALIGN_Y + (ui.Y_ELEMENT_OFFSET * 4),
                                   height=100, width=200)
        self.input_width_label.place(x=ui.ALIGN_X,
                                     y=ui.ALIGN_Y + (ui.Y_ELEMENT_OFFSET * 7),
                                     width=100)
        self.input_width.place(x=ui.ALIGN_X + ui.LABEL_OFFSET + ui.X_100_UNIT_WIDTH_OFFSET,
                               y=ui.ALIGN_Y + (ui.Y_ELEMENT_OFFSET * 7),
                               height=25, width=30)
        self.output_width_label.place(x=ui.ALIGN_X + ui.LABEL_OFFSET + ui.X_ELEMENT_OFFSET
                                        + ui.OUTPUT_WIDTH_ALIGNMENT_OFFSET
                                        + ui.X_30_UNIT_WIDTH_OFFSET
                                        + ui.X_100_UNIT_WIDTH_OFFSET,
                                      y=ui.ALIGN_Y + (ui.Y_ELEMENT_OFFSET * 7),
                                      width=100)
        self.output_width.place(x=ui.ALIGN_X + (ui.LABEL_OFFSET * 2) + ui.X_ELEMENT_OFFSET + ui.OUTPUT_WIDTH_ALIGNMENT_OFFSET
                                  + ui.X_30_UNIT_WIDTH_OFFSET
                                  + (ui.X_100_UNIT_WIDTH_OFFSET * 2),
                                y=ui.ALIGN_Y + (ui.Y_ELEMENT_OFFSET * 7),
                                height=25, width=30)
        self.stride_label.place(x=ui.ALIGN_X + (ui.LABEL_OFFSET * 2) + (ui.X_ELEMENT_OFFSET * 2)
                                  + ui.STRIDE_ALIGNMENT_OFFSET
                                  + (ui.X_30_UNIT_WIDTH_OFFSET * 2)
                                  + (ui.X_100_UNIT_WIDTH_OFFSET * 2),
                                y=ui.ALIGN_Y + (ui.Y_ELEMENT_OFFSET * 7),
                                width=100)
        self.stride.place(x=ui.ALIGN_X + (ui.LABEL_OFFSET * 3) + (ui.X_ELEMENT_OFFSET * 2)
                            + ui.STRIDE_ALIGNMENT_OFFSET
                            + (ui.X_30_UNIT_WIDTH_OFFSET * 2)
                            + (ui.X_100_UNIT_WIDTH_OFFSET * 3),
                          y=ui.ALIGN_Y + (ui.Y_ELEMENT_OFFSET * 7),
                          height=25, width=30)
        self.time_offset_label.place(x=ui.ALIGN_X + (ui.LABEL_OFFSET * 3) + (ui.X_ELEMENT_OFFSET * 3)
                                       + ui.TIME_OFFSET_ALIGNMENT_OFFSET
                                       + (ui.X_30_UNIT_WIDTH_OFFSET * 3)
                                       + (ui.X_100_UNIT_WIDTH_OFFSET * 3),
                                     y=ui.ALIGN_Y + (ui.Y_ELEMENT_OFFSET * 7),
                                     width=100)
        self.time_offset.place(x=ui.ALIGN_X + (ui.LABEL_OFFSET * 4) + (ui.X_ELEMENT_OFFSET * 3)
                                 + ui.TIME_OFFSET_ALIGNMENT_OFFSET
                                 + (ui.X_30_UNIT_WIDTH_OFFSET * 3)
                                 + (ui.X_100_UNIT_WIDTH_OFFSET * 4),
                               y=ui.ALIGN_Y + (ui.Y_ELEMENT_OFFSET * 7),
                               height=25, width=30)
        self.epoch_label.place(x=ui.ALIGN_X + (ui.LABEL_OFFSET * 4) + (ui.X_ELEMENT_OFFSET * 4)
                                 + ui.EPOCH_LABEL_ALIGNMENT_OFFSET
                                 + (ui.X_30_UNIT_WIDTH_OFFSET * 4)
                                 + (ui.X_100_UNIT_WIDTH_OFFSET * 4),
                               y=ui.ALIGN_Y + (ui.Y_ELEMENT_OFFSET * 7),
                               width=50)
        self.epoch.place(x=ui.ALIGN_X + (ui.LABEL_OFFSET * 5) + (ui.X_ELEMENT_OFFSET * 4)
                           + (ui.X_30_UNIT_WIDTH_OFFSET * 4)
                           + (ui.X_100_UNIT_WIDTH_OFFSET * 5),
                         y=ui.ALIGN_Y + (ui.Y_ELEMENT_OFFSET * 7),
                         height=25, width=75)
        self.learning_rate_label.place(x=ui.ALIGN_X + (ui.LABEL_OFFSET * 5) + (ui.X_ELEMENT_OFFSET * 5)
                                         + (ui.X_30_UNIT_WIDTH_OFFSET * 4)
                                         + (ui.X_100_UNIT_WIDTH_OFFSET * 5)
                                         + ui.X_75_UNIT_WIDTH_OFFSET,
                                       y=ui.ALIGN_Y + (ui.Y_ELEMENT_OFFSET * 7),
                                       width=100)
        self.learning_rate.place(x=ui.ALIGN_X + (ui.LABEL_OFFSET * 6) + (ui.X_ELEMENT_OFFSET * 5)
                                   + (ui.X_30_UNIT_WIDTH_OFFSET * 4)
                                   + (ui.X_100_UNIT_WIDTH_OFFSET * 6)
                                   + ui.X_75_UNIT_WIDTH_OFFSET,
                                 y=ui.ALIGN_Y + (ui.Y_ELEMENT_OFFSET * 7),
                                 height=25, width=75)
        self.train_model_button.place(x=ui.ALIGN_X + ui.LEFT_BUTTON_ALIGNMENT_OFFSET,
                                      y=ui.ALIGN_Y + (ui.Y_ELEMENT_OFFSET * 8),
                                      width=150)
        self.save_model_button.place(x=ui.ALIGN_X + ui.RIGHT_BUTTON_ALIGNMENT_OFFSET,
                                     y=ui.ALIGN_Y + (ui.Y_ELEMENT_OFFSET * 8),
                                     width=150)

        self.output_text_label.place(x=ui.ALIGN_X + 5)
        self.output_text.draw(x=ui.ALIGN_X + 10, relx=.05, rely=.05, relwidth=.9, relheight=.9)
        self.draw_split_type_inputs(self.split_type_options[0])

    def hide(self):
        MenuWindow.hide(self)
        self.input_frame.destroy()
        self.output_frame.destroy()
        self.csv_selector.destroy()
        self.schema_selector.destroy()
        self.imputer_selector_label.destroy()
        self.imputer_selector.destroy()
        self.split_type_selector_label.destroy()
        self.split_type_selector.destroy()
        self.normalization_selector_label.destroy()
        self.normalization_selector.destroy()
        self.training_features_label.destroy()
        self.training_features.destroy()
        self.output_features_label.destroy()
        self.output_features.destroy()
        self.epoch_label.destroy()
        self.epoch.destroy()
        self.learning_rate_label.destroy()
        self.learning_rate.destroy()
        self.input_width_label.destroy()
        self.input_width.destroy()
        self.output_width_label.destroy()
        self.output_width.destroy()
        self.stride_label.destroy()
        self.stride.destroy()
        self.time_offset_label.destroy()
        self.time_offset.destroy()
        self.train_model_button.destroy()
        self.save_model_button.destroy()
        self.output_text_label.destroy()
        self.output_text.hide()
        self.straight_validation_slider.destroy()
        self.straight_training_slider.destroy()
        self.rolling_gap_size.destroy()
        self.rolling_testing_size.destroy()
        self.rolling_validation_size.destroy()
        self.rolling_training_size.destroy()
        self.rolling_stride_size.destroy()
        self.expanding_gap_size.destroy()
        self.expanding_training_size.destroy()
        self.expanding_validation_size.destroy()
        self.expanding_testing_size.destroy()
        self.expanding_expansion_rate.destroy()
        self.straight_validation_slider_label.destroy()
        self.straight_training_slider_label.destroy()
        self.rolling_gap_size_label.destroy()
        self.rolling_testing_size_label.destroy()
        self.rolling_validation_size_label.destroy()
        self.rolling_training_size_label.destroy()
        self.rolling_stride_size_label.destroy()
        self.expanding_gap_size_label.destroy()
        self.expanding_training_size_label.destroy()
        self.expanding_validation_size_label.destroy()
        self.expanding_testing_size_label.destroy()
        self.expanding_expansion_rate_label.destroy()

    def draw_split_type_inputs(self, selection):
        """
        Author: Marcus Kline
        Purpose: Depending on which type of split type is selected, a different set of input data will need to be
        collected. This function will clear the frame of any existing input fields populated by this function and
        then populate specific input fields based on the selected split type.
        :param selection: Automatically filled with the selection made when split type OptionMenu is changed
        :return:
        """
        self.straight_validation_slider.place_forget()
        self.straight_training_slider.place_forget()
        self.rolling_gap_size.place_forget()
        self.rolling_testing_size.place_forget()
        self.rolling_validation_size.place_forget()
        self.rolling_training_size.place_forget()
        self.rolling_stride_size.place_forget()
        self.expanding_gap_size.place_forget()
        self.expanding_training_size.place_forget()
        self.expanding_validation_size.place_forget()
        self.expanding_testing_size.place_forget()
        self.expanding_expansion_rate.place_forget()
        self.straight_validation_slider_label.place_forget()
        self.straight_training_slider_label.place_forget()
        self.rolling_gap_size_label.place_forget()
        self.rolling_testing_size_label.place_forget()
        self.rolling_validation_size_label.place_forget()
        self.rolling_training_size_label.place_forget()
        self.rolling_stride_size_label.place_forget()
        self.expanding_gap_size_label.place_forget()
        self.expanding_training_size_label.place_forget()
        self.expanding_validation_size_label.place_forget()
        self.expanding_testing_size_label.place_forget()
        self.expanding_expansion_rate_label.place_forget()
        if selection == self.split_type_options[0]:
            self.straight_training_slider_label.place(x=ui.ALIGN_X,
                                                      y=ui.ALIGN_Y + (ui.Y_ELEMENT_OFFSET * 2),
                                                      width=100)
            self.straight_training_slider.place(x=ui.ALIGN_X + ui.LABEL_OFFSET + ui.X_100_UNIT_WIDTH_OFFSET,
                                                y=ui.ALIGN_Y + (ui.Y_ELEMENT_OFFSET * 2),
                                                width=200, height=40)
            self.straight_validation_slider_label.place(x=ui.ALIGN_X + ui.LABEL_OFFSET + ui.X_ELEMENT_OFFSET
                                                          + ui.X_100_UNIT_WIDTH_OFFSET
                                                          + ui.X_200_UNIT_WIDTH_OFFSET,
                                                        y=ui.ALIGN_Y + (ui.Y_ELEMENT_OFFSET * 2),
                                                        width=100)
            self.straight_validation_slider.place(x=ui.ALIGN_X + (ui.LABEL_OFFSET * 2) + ui.X_ELEMENT_OFFSET
                                                    + (ui.X_100_UNIT_WIDTH_OFFSET * 2)
                                                    + ui.X_200_UNIT_WIDTH_OFFSET,
                                                  y=ui.ALIGN_Y + (ui.Y_ELEMENT_OFFSET * 2),
                                                  width=200, height=40)
        if selection == self.split_type_options[1]:
            self.rolling_training_size_label.place(x=ui.ALIGN_X,
                                                   y=ui.ALIGN_Y + (ui.Y_ELEMENT_OFFSET * 2),
                                                   width=100)
            self.rolling_training_size.place(x=ui.ALIGN_X + ui.LABEL_OFFSET + ui.X_100_UNIT_WIDTH_OFFSET,
                                             y=ui.ALIGN_Y + (ui.Y_ELEMENT_OFFSET * 2),
                                             width=200, height=25)
            self.rolling_validation_size_label.place(x=ui.ALIGN_X + ui.LABEL_OFFSET + ui.X_ELEMENT_OFFSET
                                                       + ui.X_100_UNIT_WIDTH_OFFSET
                                                       + ui.X_200_UNIT_WIDTH_OFFSET,
                                                     y=ui.ALIGN_Y + (ui.Y_ELEMENT_OFFSET * 2),
                                                     width=100)
            self.rolling_validation_size.place(x=ui.ALIGN_X + (ui.LABEL_OFFSET * 2) + ui.X_ELEMENT_OFFSET
                                                 + (ui.X_100_UNIT_WIDTH_OFFSET * 2)
                                                 + ui.X_200_UNIT_WIDTH_OFFSET,
                                               y=ui.ALIGN_Y + (ui.Y_ELEMENT_OFFSET * 2),
                                               width=200, height=25)
            self.rolling_testing_size_label.place(x=ui.ALIGN_X,
                                                  y=ui.ALIGN_Y + (ui.Y_ELEMENT_OFFSET * 3),
                                                  width=100)
            self.rolling_testing_size.place(x=ui.ALIGN_X + ui.LABEL_OFFSET + ui.X_100_UNIT_WIDTH_OFFSET,
                                            y=ui.ALIGN_Y + (ui.Y_ELEMENT_OFFSET * 3),
                                            width=200, height=25)
            self.rolling_gap_size_label.place(x=ui.ALIGN_X + ui.LABEL_OFFSET + ui.X_ELEMENT_OFFSET
                                                + ui.X_100_UNIT_WIDTH_OFFSET
                                                + ui.X_200_UNIT_WIDTH_OFFSET,
                                              y=ui.ALIGN_Y + (ui.Y_ELEMENT_OFFSET * 3),
                                              width=100)
            self.rolling_gap_size.place(x=ui.ALIGN_X + (ui.LABEL_OFFSET * 2) + ui.X_ELEMENT_OFFSET
                                          + (ui.X_100_UNIT_WIDTH_OFFSET * 2)
                                          + ui.X_200_UNIT_WIDTH_OFFSET,
                                        y=ui.ALIGN_Y + (ui.Y_ELEMENT_OFFSET * 3),
                                        width=200, height=25)
            self.rolling_stride_size_label.place(x=ui.ALIGN_X + (ui.LABEL_OFFSET * 2) + (ui.X_ELEMENT_OFFSET * 2)
                                                   + (ui.X_100_UNIT_WIDTH_OFFSET * 2)
                                                   + (ui.X_200_UNIT_WIDTH_OFFSET * 2),
                                                 y=ui.ALIGN_Y + (ui.Y_ELEMENT_OFFSET * 3),
                                                 width=100)
            self.rolling_stride_size.place(x=ui.ALIGN_X + (ui.LABEL_OFFSET * 3) + (ui.X_ELEMENT_OFFSET * 2)
                                             + (ui.X_100_UNIT_WIDTH_OFFSET * 3)
                                             + (ui.X_200_UNIT_WIDTH_OFFSET * 2),
                                           y=ui.ALIGN_Y + (ui.Y_ELEMENT_OFFSET * 3),
                                           width=200, height=25)
        if selection == self.split_type_options[2]:
            self.expanding_training_size_label.place(x=ui.ALIGN_X,
                                                     y=ui.ALIGN_Y + (ui.Y_ELEMENT_OFFSET * 2),
                                                     width=100)
            self.expanding_training_size.place(x=ui.ALIGN_X + ui.LABEL_OFFSET
                                                 + ui.X_100_UNIT_WIDTH_OFFSET,
                                               y=ui.ALIGN_Y + (ui.Y_ELEMENT_OFFSET * 2),
                                               width=200, height=25)
            self.expanding_validation_size_label.place(x=ui.ALIGN_X + ui.LABEL_OFFSET + ui.X_ELEMENT_OFFSET
                                                         + ui.X_100_UNIT_WIDTH_OFFSET
                                                         + ui.X_200_UNIT_WIDTH_OFFSET,
                                                       y=ui.ALIGN_Y + (ui.Y_ELEMENT_OFFSET * 2),
                                                       width=100)
            self.expanding_validation_size.place(x=ui.ALIGN_X + (ui.LABEL_OFFSET * 2) + ui.X_ELEMENT_OFFSET
                                                   + (ui.X_100_UNIT_WIDTH_OFFSET * 2)
                                                   + ui.X_200_UNIT_WIDTH_OFFSET,
                                                 y=ui.ALIGN_Y + (ui.Y_ELEMENT_OFFSET * 2),
                                                 width=200, height=25)
            self.expanding_testing_size_label.place(x=ui.ALIGN_X,
                                                    y=ui.ALIGN_Y + (ui.Y_ELEMENT_OFFSET * 3),
                                                    width=100)
            self.expanding_testing_size.place(x=ui.ALIGN_X + ui.LABEL_OFFSET + ui.X_100_UNIT_WIDTH_OFFSET,
                                              y=ui.ALIGN_Y + (ui.Y_ELEMENT_OFFSET * 3),
                                              width=200, height=25)
            self.expanding_gap_size_label.place(x=ui.ALIGN_X + ui.LABEL_OFFSET + ui.X_ELEMENT_OFFSET
                                                  + ui.X_100_UNIT_WIDTH_OFFSET
                                                  + ui.X_200_UNIT_WIDTH_OFFSET,
                                                y=ui.ALIGN_Y + (ui.Y_ELEMENT_OFFSET * 3),
                                                width=100)
            self.expanding_gap_size.place(x=ui.ALIGN_X + (ui.LABEL_OFFSET * 2) + ui.X_ELEMENT_OFFSET
                                            + (ui.X_100_UNIT_WIDTH_OFFSET * 2)
                                            + ui.X_200_UNIT_WIDTH_OFFSET,
                                          y=ui.ALIGN_Y + (ui.Y_ELEMENT_OFFSET * 3),
                                          width=200, height=25)
            self.expanding_expansion_rate_label.place(x=ui.ALIGN_X + (ui.LABEL_OFFSET * 2) + (ui.X_ELEMENT_OFFSET * 2)
                                                        + (ui.X_100_UNIT_WIDTH_OFFSET * 2)
                                                        + (ui.X_200_UNIT_WIDTH_OFFSET * 2),
                                                      y=ui.ALIGN_Y + (ui.Y_ELEMENT_OFFSET * 3),
                                                      width=100)
            self.expanding_expansion_rate.place(x=ui.ALIGN_X + (ui.LABEL_OFFSET * 3) + (ui.X_ELEMENT_OFFSET * 2)
                                                  + (ui.X_100_UNIT_WIDTH_OFFSET * 3)
                                                  + (ui.X_200_UNIT_WIDTH_OFFSET * 2),
                                                y=ui.ALIGN_Y + (ui.Y_ELEMENT_OFFSET * 3),
                                                width=200, height=25)

    def save_model(self):
        """
        Author: Alexander Cherry
        Purpose: This function is used to save the trained model using the .h5 file type. A file explorer dialog opens
        to select the location where the trained model will be saved.
        :return:
        """
        if self.trained_model is None:
            self.output_text.output('No model has been trained yet.')
            return
        save_loc = fdiag.asksaveasfilename(defaultextension=(ui.MODEL_FILE_LABEL, ui.MODEL_FILE_TYPE),
                                           filetypes=[(ui.MODEL_FILE_LABEL, ui.MODEL_FILE_TYPE)])
        if save_loc != '':
            save_loc = save_loc if save_loc.endswith('.h5') else save_loc + '.h5'
            self.trained_model.save(save_loc)
            self.model_fit_reporter.save(save_loc[:-3])
            self.trained_model = None

    def train_model(self):
        """
        Author: Alexander Cherry
        Purpose: This function is used to train the model using values in the input fields under the TrainMenu
        :return:
        """
        if self.training_csv is None:
            self.output_text.output('No CSV file has been selected as training data.')
            return
        imputer = self.imputer_selection.get()
        imputed_data = pipeline.DataImputer(imputer)(self.training_csv)
        split = self.split_type_selection.get()
        splits = None
        if split == 'Straight Split':
            train_size = self.straight_training_slider.get() / 100
            val_size = self.straight_validation_slider.get() / 100
            splits = pipeline.StraightSplit(train_size, val_size)(imputed_data)
        elif split == 'Rolling Split':
            train_size = self.rolling_training_size.get()
            test_size = self.rolling_testing_size.get()
            val_size = self.rolling_validation_size.get()
            stride = self.rolling_stride_size.get()
            gap = self.rolling_gap_size.get()
            splits = pipeline.RollingSplit(train_size, test_size, val_size, stride, gap)(imputed_data)
        elif split == 'Expanding Split':
            train_size = self.expanding_training_size.get()
            test_size = self.expanding_testing_size.get()
            val_size = self.expanding_validation_size.get()
            expansion_rate = self.expanding_expansion_rate.get()
            gap = self.expanding_gap_size.get()
            splits = pipeline.ExpandingSplit(train_size, test_size, val_size, expansion_rate, gap)(imputed_data)
        normalizer = self.normalization_selection.get()
        normalized_splits = None
        if normalizer == 'Min-Max':
            normalized_splits = pipeline.MinMaxNormalizer(splits)()
        elif normalizer == 'Z Standardization':
            normalized_splits = pipeline.ZStandardizer(splits)()
        features_in = [self.training_features.get(i) for i in self.training_features.curselection()]
        features_out = [self.output_features.get(i) for i in self.output_features.curselection()]
        width_in = int(self.input_width.get("1.0", "end-1c"))
        width_out = int(self.output_width.get("1.0", "end-1c"))
        transformer_stride = int(self.stride.get("1.0", "end-1c"))
        time_offset = int(self.time_offset.get("1.0", "end-1c"))
        timeseries_data = pipeline.SupervisedTimeseriesTransformer(features_in,
                                                                   features_out,
                                                                   width_in,
                                                                   width_out,
                                                                   transformer_stride,
                                                                   time_offset)(normalized_splits)
        model_path = self.path_to_model_schema
        epochs = int(self.epoch.get("1.0", "end-1c"))
        learning_rate = float(self.learning_rate.get("1.0", "end-1c"))
        self.trained_model = pipeline.ForecastModelTrainer(model_path)(
            timeseries_data,
            epochs,
            learning_rate,
            callbacks=[OutputEpoch(self.output_text, epochs)]
        )
        self.model_fit_reporter = ModelEvaluationReporter(self.trained_model)
        report = self.model_fit_reporter(timeseries_data)
        self.output_text.append_output(f'Your model has finished training!\n'
                                       f'Press the "Save Model" button to save it as a file.\n'
                                       f'---------------------------------------------------\n'
                                       f'Model Performance:\n'
                                       f'{report}')

    def generate_training_and_output_features(self):
        """
        Author: Marcus Kline
        Purpose: This function is used to populate the training_features and output_features ListBoxes with the columns
        from the uploaded csv (if one has been uploaded) for a user to select from.
        :return:
        """
        if self.training_csv is not None:
            self.training_features.delete(0, tk.END)
            self.output_features.delete(0, tk.END)
            for i in self.training_csv:
                self.training_features.insert(tk.END, self.training_csv[i].name)
            for i in self.training_csv:
                self.output_features.insert(tk.END, self.training_csv[i].name)

    def upload_csv(self):
        """
        Author:Marcus Kline
        Purpose: This function is used to upload a csv which is then used to populate the training_features and
        output_features ListBoxes
        :return:
        """
        self.path_to_csv = fdiag.askopenfilename(filetypes=[(ui.CSV_FILE_LABEL, ui.CSV_FILE_TYPE)])
        if self.path_to_csv is not None:
            df = pd.read_csv(self.path_to_csv, parse_dates=True)
            self.training_csv = df.select_dtypes(include=[np.number])
            self.generate_training_and_output_features()
            self.csv_selection_label.config(text=os.path.basename(self.path_to_csv))

    def upload_schema(self):
        """
        Author: Marcus Kline
        Purpose: This function is used to upload a schema in the form of a JSON file. This is then used in the
        train_model function
        :return: 
        """
        self.path_to_model_schema = fdiag.askopenfilename(filetypes=[(ui.JSON_FILE_LABEL, ui.JSON_FILE_TYPE)])
        self.schema_selection_label.config(text=os.path.basename(self.path_to_model_schema))


class ClimateChangeMenu(MenuWindow):
    def __init__(self, app_frame: tk.Frame):
        MenuWindow.__init__(self, app_frame)
        self.dates = []
        self.generate_plot_button = None  # Type: tk.Button
        self.data_source_table_button = None  # Type: tk.Button
        self.upload_new_data = None  # Type: tk.Button
        self.graph_frame = None  # Type: tk.Frame
        self.tool_frame = None  # Type: tk.Frame
        self.data_selection_box = None  # Type: tk.Listbox
        self.source_data = None  # Type: pd.DataFrame

    def init_ui(self):
        MenuWindow.init_ui(self)
        try:
            self.source_data = ClimateAccess.get_source_data()
        except FileNotFoundError:
            print('Unable to load Default Data')
        self.graph_frame = tk.Frame(master=self.body, bg=ui.BACKGROUND_COLOR)
        self.tool_frame = tk.Frame(master=self.body, bg=ui.BACKGROUND_COLOR)
        self.upload_new_data = tk.Button(
            self.tool_frame,
            text="Upload New Data",
            bg=ui.BUTTON_BACKGROUND,
            fg=ui.BUTTON_FOREGROUND,
            command=lambda: self.upload_data()
        )
        self.generate_plot_button = tk.Button(
            self.tool_frame,
            text="Generate Line Graph",
            bg=ui.BUTTON_BACKGROUND,
            fg=ui.BUTTON_FOREGROUND,
            command=lambda: self.generate_plot()
        )
        self.data_source_table_button = tk.Button(
            self.tool_frame,
            text="Show Source Data Explorer",
            bg=ui.BUTTON_BACKGROUND,
            fg=ui.BUTTON_FOREGROUND,
            command=lambda: self.generate_table()  # Todo: add functionality for this button
        )
        self.data_selection_box = tk.Listbox(
            self.tool_frame,
            bg=ui.BUTTON_BACKGROUND,
            fg=ui.BUTTON_FOREGROUND,
            selectmode=tk.EXTENDED
        )

    def draw(self):
        MenuWindow.draw(self)
        self.graph_frame.place(relx=.2, rely=0, relwidth=0.8, relheight=1)
        self.tool_frame.place(relx=0, rely=0, relwidth=.2, relheight=1)
        self.generate_plot_button.place(x=ui.ALIGN_X + 5, y=ui.ALIGN_Y + 5, relwidth=.9)
        self.data_source_table_button.place(x=ui.ALIGN_X + 5, y=ui.ALIGN_Y + 60, relwidth=.9)
        self.data_selection_box.place(x=ui.ALIGN_X + 5, y=ui.ALIGN_Y + 125, relwidth=.9, relheight=.6)
        self.upload_new_data.place(x=ui.ALIGN_X + 5, rely=.9, relwidth=.9)
        self.populate_data_picker()

    def hide(self):
        MenuWindow.hide(self)

    def generate_plot(self):
        """
        Author: Marcus Kline
        Purpose: plots uploaded data on a 2D line graph. If no data is uploaded then
                    default data is used instead. This data is generated from https://www.esrl.noaa.gov/gmd/dv/data/
                    and is included in \\WeatherAI\\data\\mlo_full.csv
        :return:
        """
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
        figure = Figure(figsize=(7, 5), dpi=100, constrained_layout=True)
        plot1 = figure.add_subplot(111)
        plot1.grid()
        selected_data = [self.data_selection_box.get(i) for i in self.data_selection_box.curselection()]
        plot1.set_ylabel(','.join(map(str, selected_data)))
        plot1.set_xlabel('Dates')
        for field in selected_data:
            plot1.plot(self.dates, self.source_data[field])
        plot1.legend(selected_data, loc='upper right')
        canvas = FigureCanvasTkAgg(figure, master=self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()
        toolbar = NavigationToolbar2Tk(canvas, self.graph_frame)
        toolbar.update()

    def populate_data_picker(self):
        """
        Author: Marcus Kline
        Purpose: Populates the data picker text field with columns from the data source.
        :return:
        """
        self.data_selection_box.delete(0, tk.END)
        self.dates = self.source_data[['date']]
        for i in self.source_data:
            self.data_selection_box.insert(tk.END, self.source_data[i].name)
        self.data_selection_box.delete(0)

    def generate_table(self):
        """
        Author: Marcus Kline
        Purpose: generates a table of the uploaded data on a 2D line graph. If no data is uploaded then
                        default data is used instead. This data is generated from https://www.esrl.noaa.gov/gmd/dv/data/
                        and is included in \\WeatherAI\\data\\mlo_full.csv
        :return:
        """
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
        table = Table(self.graph_frame, dataframe=self.source_data, showtoolbar=True, showstatusbar=True)
        table.show()
        table.redraw()

    def upload_data(self):
        """
        Author: Marcus Kline
        Purpose: This function is used to upload a csv file to be used for plotting its data, displaying it in a
        table, and populating the data_viewer listbox
        :return: 
        """
        new_data = fdiag.askopenfile(mode='r', filetypes=[(ui.CSV_FILE_LABEL, ui.CSV_FILE_TYPE)])
        if new_data is not None:
            self.source_data = pd.read_csv(new_data, parse_dates=True)
            self.populate_data_picker()
