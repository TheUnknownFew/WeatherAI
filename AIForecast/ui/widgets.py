import time
import tkinter as tk
from enum import Enum
from threading import Thread
from tkinter.filedialog import asksaveasfile
from typing import List, Dict

from AIForecast import utils
from AIForecast.RNN.WeatherForecasting import ForecastingNetwork
from AIForecast.access import ClimateAccess
from AIForecast.access import WeatherAccess
from AIForecast.utils import DataUtils

import pandas as pd
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
from pandastable import Table, TableModel

BACKGROUND_COLOR = '#525453'
"""
Default background color for most UI elements.
"""

FOREGROUND_COLOR = 'white'

H1_FONT = 100

BUTTON_BORDER_WIDTH = 1
BUTTON_BACKGROUND = 'white'
BUTTON_ACTIVE_BACKGROUND = 'gray'
BUTTON_FOREGROUND = 'black'

_ALIGN_X = 10
_ALIGN_Y = 10

CSV_FILE_TYPE = '*.csv'
CSV_FILE_LABEL = 'CSV Files'


class Menus(Enum):
    """
    Menus is an enum containing named menus that exist in the application.

    Author: Alexander Cherry
    """

    MAIN_MENU = 0
    TEST_MENU = 1
    TRAIN_MENU = 2
    DATA_VIEWER_MENU = 3


class Drawable:

    def init_ui(self):
        """
        Init_ui is an abstract function.
        Use init_ui to instantiate ui elements before drawing them.

        Author: Alexander Cherry
        """
        pass

    def draw(self):
        """
        Draw is an abstract function.
        Use draw to show elements of the inheriting class.
        It is recommended to invoke the super draw method before drawing additional elements.

        Author: Alexander Cherry
        """
        pass

    def hide(self):
        """
        Hide is an abstract function.
        Use hide to destroy all elements that were added to the screen by the draw method.
        It is recommended to invoke the super hide method to ensure all nested elements are remove.

        Author: Alexander Cherry
        """
        pass


class NavBar(Drawable):
    NAV_COLOR = 'gray'
    NAV_HEIGHT = 40

    def __init__(self, parent: tk.Frame):
        self.parent = parent
        self.nav_buttons: List[tk.Button] = []
        self.nav_frame = None

    def init_ui(self):
        self.nav_frame = tk.Frame(self.parent, bg=self.NAV_COLOR)
        self.nav_buttons.append(
            tk.Button(
                self.nav_frame,
                text="Main",
                borderwidth=BUTTON_BORDER_WIDTH,
                bg=BUTTON_BACKGROUND,
                fg=BUTTON_FOREGROUND,
                command=lambda: AppWindow.display_screen(Menus.MAIN_MENU)
            )
        )
        self.nav_buttons.append(
            tk.Button(
                self.nav_frame,
                text="Test",
                borderwidth=BUTTON_BORDER_WIDTH,
                bg=BUTTON_BACKGROUND,
                fg=BUTTON_FOREGROUND,
                command=lambda: AppWindow.display_screen(Menus.TEST_MENU)
            )
        )
        self.nav_buttons.append(
            tk.Button(
                self.nav_frame,
                text="Train",
                borderwidth=BUTTON_BORDER_WIDTH,
                bg=BUTTON_BACKGROUND,
                fg=BUTTON_FOREGROUND,
                command=lambda: AppWindow.display_screen(Menus.TRAIN_MENU)
            )
        )
        self.nav_buttons.append(
            tk.Button(
                self.nav_frame,
                text="Data Viewer",
                borderwidth=BUTTON_BORDER_WIDTH,
                bg=BUTTON_BACKGROUND,
                fg=BUTTON_FOREGROUND,
                command=lambda: AppWindow.display_screen(Menus.DATA_VIEWER_MENU)
            )
        )

    def draw(self):
        self.nav_frame.place(x=0, y=0, relwidth=1, height=self.NAV_HEIGHT)
        i = 0
        for button in self.nav_buttons:
            button.place(x=10 + 100 * i + 5 * i, rely=0.2, relheight=0.55, width=100)
            i += 1

    def hide(self):
        self.nav_frame.destroy()
        for button in self.nav_buttons:
            button.destroy()
        self.nav_buttons.clear()


class Menu(Drawable):
    """
    A standard Menu has a Nav bar and a body.
    The body of the Menu is initially blank.
    Classes extending Menu are tasked with filling the body
    with content.

    Author: Alexander Cherry
    """

    def __init__(self, app_frame: tk.Frame):
        self.container = app_frame
        self.body_height = 0
        self.nav_bar = NavBar(self.container)
        self.body = None

    def init_ui(self):
        self.nav_bar.init_ui()
        self.body = tk.Frame(self.container, bg=BACKGROUND_COLOR)

    def draw(self):
        self.nav_bar.draw()
        self.body_height = AppWindow.current_height - NavBar.NAV_HEIGHT
        self.body.place(x=0, y=NavBar.NAV_HEIGHT, relwidth=1, height=self.body_height)

    def hide(self):
        self.nav_bar.hide()
        self.body.destroy()


class SplitWindowMenu(Menu):
    def __init__(self, app_frame: tk.Frame):
        super().__init__(app_frame)
        self.top: tk.Frame = None
        self.bottom: tk.Frame = None

    def init_ui(self):
        super().init_ui()
        self.top = tk.Frame(self.body, bg=BACKGROUND_COLOR)
        self.bottom = tk.Frame(self.body, bg=BACKGROUND_COLOR)

    def draw(self):
        super().draw()
        self.top.place(relx=0, rely=0, relwidth=1, relheight=0.5)
        self.bottom.place(relx=0, rely=0.5, relwidth=1, relheight=0.5)

    def hide(self):
        super().hide()
        self.top.destroy()
        self.bottom.destroy()


class IOMenu(SplitWindowMenu):
    _INPUT_LABEL = "Choose Input Source:"
    _CITY_SELECT_LABEL = "Choose surrounding cities from the list:"
    _TARGET_SELECT_LABEL = "Choose a target city from the list:"

    def __init__(self, app_frame: tk.Frame, output_label: str = 'Label not set'):
        super().__init__(app_frame)
        self.left_pane: tk.Frame = None
        self.right_pane: tk.Frame = None
        # Input frame elements:
        # Right pane:
        self.source_select_label: tk.Label = None
        # Left pane:
        self.surrounding_cities_label: tk.Label = None
        self.surrounding_cities_select: tk.Listbox = None
        self.target_city_var = tk.StringVar()
        self.target_city_label: tk.Label = None
        self.target_city_select: tk.OptionMenu = None
        # Output frame elements:
        self.output_label_text = output_label
        self.output_label: tk.Label = None
        self.button_execute: tk.Button = None
        self.output_window: tk.Text = None

    def _populate_surrounding_cities_listbox(self):
        for city in WeatherAccess.cities:
            self.surrounding_cities_select.insert(tk.END, city)

    def init_ui(self):
        super().init_ui()
        self.left_pane = tk.Frame(self.top, bg=BACKGROUND_COLOR)
        self.right_pane = tk.Frame(self.top, bg=BACKGROUND_COLOR)
        self._init_input_frame()
        self._init_output_frame()

    def _init_input_frame(self):
        self.source_select_label = tk.Label(
            self.left_pane,
            text=self._INPUT_LABEL,
            bg=BACKGROUND_COLOR,
            fg=FOREGROUND_COLOR,
            font=H1_FONT
        )
        self.surrounding_cities_label = tk.Label(
            self.right_pane,
            text=self._CITY_SELECT_LABEL,
            bg=BACKGROUND_COLOR,
            fg=FOREGROUND_COLOR,
            font=H1_FONT
        )
        self.surrounding_cities_select = tk.Listbox(self.right_pane, selectmode="extended")
        self._populate_surrounding_cities_listbox()
        self.target_city_label = tk.Label(
            self.right_pane,
            text=self._TARGET_SELECT_LABEL,
            bg=BACKGROUND_COLOR,
            fg=FOREGROUND_COLOR,
            font=H1_FONT
        )
        self.target_city_select = tk.OptionMenu(self.right_pane, self.target_city_var, *WeatherAccess.cities)
        self.target_city_var.set(next(iter(WeatherAccess.get_cities())))

    def _init_output_frame(self):
        self.output_label = tk.Label(self.bottom, text=self.output_label_text, bg=BACKGROUND_COLOR, fg=FOREGROUND_COLOR)
        self.output_label.config(font=H1_FONT)
        self.button_execute = tk.Button(self.bottom, text="Run", borderwidth=BUTTON_BORDER_WIDTH)
        self.button_execute.config(command=lambda: utils.log(__name__).debug("Function for button has not been set!"))
        self.output_window = tk.Text(self.bottom)
        self.output("Waiting for input!")

    def draw(self):
        super().draw()
        self.left_pane.place(relx=0, rely=0, relwidth=0.3, relheight=1)
        self.right_pane.place(relx=0.3, rely=0, relwidth=0.7, relheight=1)
        self._draw_input_frame()
        self._draw_output_frame()

    def _draw_input_frame(self):
        self.source_select_label.place(x=_ALIGN_X, y=_ALIGN_Y)
        self.surrounding_cities_label.place(x=_ALIGN_X, y=_ALIGN_Y)
        label_height = self.get_label_height()
        label_width = self.get_label_width()
        self.surrounding_cities_select.place(x=_ALIGN_X + 5, y=_ALIGN_Y + label_height, width=label_width)
        self.target_city_label.place(x=label_width + 30, y=_ALIGN_Y)
        self.target_city_select.place(x=label_width + 35, y=_ALIGN_Y + label_height)

    def _draw_output_frame(self):
        self.output_label.place(x=_ALIGN_X, y=_ALIGN_Y)
        label_height = self.output_label.winfo_reqheight()
        self.button_execute.place(x=_ALIGN_X, y=_ALIGN_Y + label_height + 5, width=80, height=40)
        self.output_window.place(x=_ALIGN_X, y=_ALIGN_Y + label_height + 50, relwidth=0.6, relheight=0.5)

    def hide(self):
        super().hide()
        self.left_pane.destroy()
        self.right_pane.destroy()
        self.source_select_label.destroy()
        self.surrounding_cities_label.destroy()
        self.surrounding_cities_select.destroy()
        self.target_city_label.destroy()
        self.target_city_select.destroy()
        self.output_label.destroy()
        self.button_execute.destroy()
        self.output_window.destroy()

    def get_label_width(self):
        return self.surrounding_cities_label.winfo_reqwidth()

    def get_label_height(self):
        return self.surrounding_cities_label.winfo_reqheight()

    def output(self, message: str):
        self.output_window.delete("1.0", tk.END)
        self.output_window.insert(tk.INSERT, message)
        self.output_window.update()

    @property
    def selected_cities(self):
        return [city for city in
                [self.surrounding_cities_select.get(i)
                 for i in self.surrounding_cities_select.curselection()]
                if city != self.target_city_var]

    @property
    def target_city(self):
        return self.target_city_var.get()


class MainMenu(Menu):
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
        self.label = tk.Label(self.body, text=self.LABEL_TEXT, bg=BACKGROUND_COLOR, fg=FOREGROUND_COLOR)

    def draw(self):
        super().draw()
        self.label.place(relx=0, rely=0, relwidth=1, relheight=0.5)

    def hide(self):
        super().hide()
        self.label.destroy()


class TestMenu(IOMenu):
    def __init__(self, app_frame: tk.Frame):
        super().__init__(app_frame)
        self.upload_trained_model_button = None
        self.trained_model = None
        self.output_frame = None
        self.input_frame = None
        self.time_horizon = None
        self.time_horizon_label = None
        self.test_button = None
        self.output_text = None
        self.output_text_label = None

    def init_ui(self):
        super().init_ui()
        self.input_frame = tk.Frame(master=self.body, bg=BACKGROUND_COLOR)
        self.output_frame = tk.Frame(master=self.body, bg=BACKGROUND_COLOR)
        self.upload_trained_model_button = tk.Button(
            self.input_frame,
            text="Upload Trained Model",
            bg=BUTTON_BACKGROUND,
            fg=BUTTON_FOREGROUND,
            command=lambda: self.upload_trained_model()
        )
        self.time_horizon_label = tk.Label(self.input_frame, text="Time Horizon (Hours)", bg=BACKGROUND_COLOR,
                                           fg=FOREGROUND_COLOR, anchor="w")
        self.time_horizon = tk.Text(self.input_frame)
        self.test_button = tk.Button(
            self.input_frame,
            text="Test",
            bg=BUTTON_BACKGROUND,
            fg=BUTTON_FOREGROUND,
            command=lambda: self.test_model()
        )
        self.output_text_label = tk.Label(self.output_frame, text="Output:", bg=BACKGROUND_COLOR, fg=FOREGROUND_COLOR,
                                          anchor="w")
        self.output_text = tk.Label(self.output_frame)

    def draw(self):
        super().draw()
        self.output_frame.place(relx=.2, rely=0, relwidth=.8, relheight=1)
        self.input_frame.place(relx=0, rely=0, relwidth=.2, relheight=1)
        self.upload_trained_model_button.place(x=_ALIGN_X + 5, y=_ALIGN_Y + 5, width=150)
        self.time_horizon_label.place(x=_ALIGN_X + 5, y=_ALIGN_Y + 45, width=150)
        self.time_horizon.place(x=_ALIGN_X + 5, y=_ALIGN_Y + 70, width=150, height=25)
        self.output_text_label.place(relx=.05)
        self.output_text.place(relx=.05, rely=.05, relwidth=.9, relheight=.9)
        self.test_button.place(x=_ALIGN_X + 5, y=_ALIGN_Y + 125, width=150)

    def hide(self):
        super().hide()

    def upload_trained_model(self):
        new_data = filedialog.askopenfile(mode='r', filetypes=[(CSV_FILE_LABEL, CSV_FILE_TYPE)])
        if new_data is not None:
            self.trained_model = pd.read_csv(new_data)

    def test_model(self):
        pass


class TrainMenu(IOMenu):
    def __init__(self, app_frame: tk.Frame):
        super().__init__(app_frame, "Press the button below to begin training:")
        self.normalization_options = ('Min-Max', 'Two Standardization')
        self.normalization_selection = tk.StringVar()
        self.split_type_options = ("Straight Split", "Rolling Split", "Expanding Split")
        self.split_type_selection = tk.StringVar()
        self.inputer_options = ("None", "Simple", "Iterative")
        self.inputer_selection = tk.StringVar()
        self.inputer_selector_label = tk.Label()
        self.csv_selector = None
        self.schema_selector = None
        self.inputer_selector = None
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
        self.expanding_gap_size = None
        self.expanding_expansion_rate = None
        self.normalization_selector = None
        self.training_features = None
        self.output_features = None
        self.input_width = None
        self.output_width = None
        self.stride = None
        self.gap_size = None
        self.epoch = None
        self.train_model_button = None
        self.output_text = None
        self.save_model_button = None
        self.input_frame = None
        self.output_frame = None
        self.training_csv = None
        self.training_schema = None
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
        self.expanding_gap_size_label = None
        self.expanding_expansion_rate_label = None
        self.training_features_label = None
        self.output_features_label = None
        self.epoch_label = None
        self.input_width_label = None
        self.output_width_label = None
        self.stride_label = None
        self.gap_size_label = None
        self.output_text_label = None
        self.learning_rate = None
        self.learning_rate_label = None

    def init_ui(self):
        super().init_ui()
        self.input_frame = tk.Frame(master=self.body, bg=BACKGROUND_COLOR)
        self.output_frame = tk.Frame(master=self.body, bg=BACKGROUND_COLOR)
        self.csv_selector = tk.Button(
            self.input_frame,
            text="Upload CSV",
            bg=BUTTON_BACKGROUND,
            fg=BUTTON_FOREGROUND,
            command=lambda: self.upload_csv()
        )
        self.schema_selector = tk.Button(
            self.input_frame,
            text="Upload Model Schema",
            bg=BUTTON_BACKGROUND,
            fg=BUTTON_FOREGROUND,
            command=lambda: self.upload_schema()
        )
        self.inputer_selection.set(self.inputer_options[0])
        self.inputer_selector = tk.OptionMenu(
            self.input_frame,
            self.inputer_selection,
            *self.inputer_options
        )
        self.inputer_selector_label = tk.Label(self.input_frame, text="Inputer Type:", bg=BACKGROUND_COLOR,
                                               fg=FOREGROUND_COLOR, anchor="e")
        self.split_type_selection.set(self.split_type_options[0])
        self.split_type_selector = tk.OptionMenu(
            self.input_frame,
            self.split_type_selection,
            *self.split_type_options,
            command=self.draw_split_type_inputs
        )
        self.split_type_selector_label = tk.Label(self.input_frame, text="Split Type:", bg=BACKGROUND_COLOR,
                                                  fg=FOREGROUND_COLOR, anchor="e")
        self.straight_training_slider = tk.Scale(self.input_frame, from_=0, to=100, tickinterval=1,
                                                 orient=tk.HORIZONTAL)
        self.straight_training_slider.set(80)
        self.straight_training_slider_label = tk.Label(self.input_frame, text="Training Percent:", bg=BACKGROUND_COLOR,
                                                       fg=FOREGROUND_COLOR, anchor="e")
        self.straight_validation_slider = tk.Scale(self.input_frame, from_=0, to=100, tickinterval=1,
                                                   orient=tk.HORIZONTAL)
        self.straight_validation_slider.set(0)
        self.straight_validation_slider_label = tk.Label(self.input_frame, text="Validation Percent:",
                                                         bg=BACKGROUND_COLOR, fg=FOREGROUND_COLOR, anchor="e")
        self.rolling_training_size = tk.Text(self.input_frame)
        self.rolling_training_size.insert(tk.END, '20')
        self.rolling_training_size_label = tk.Label(self.input_frame, text="Training Size:", bg=BACKGROUND_COLOR,
                                                    fg=FOREGROUND_COLOR, anchor="e")
        self.rolling_validation_size = tk.Text(self.input_frame)
        self.rolling_validation_size.insert(tk.END, '0')
        self.rolling_validation_size_label = tk.Label(self.input_frame, text="Validation Size:", bg=BACKGROUND_COLOR,
                                                      fg=FOREGROUND_COLOR, anchor="e")
        self.rolling_testing_size = tk.Text(self.input_frame)
        self.rolling_testing_size.insert(tk.END, '10')
        self.rolling_testing_size_label = tk.Label(self.input_frame, text="Testing Size:", bg=BACKGROUND_COLOR,
                                                   fg=FOREGROUND_COLOR, anchor="e")
        self.rolling_gap_size = tk.Text(self.input_frame)
        self.rolling_gap_size.insert(tk.END, '0')
        self.rolling_gap_size_label = tk.Label(self.input_frame, text="Gap Size:", bg=BACKGROUND_COLOR,
                                               fg=FOREGROUND_COLOR, anchor="e")
        self.rolling_stride_size = tk.Text(self.input_frame)
        self.rolling_stride_size.insert(tk.END, '1')
        self.rolling_stride_size_label = tk.Label(self.input_frame, text="Stride Size:", bg=BACKGROUND_COLOR,
                                                  fg=FOREGROUND_COLOR, anchor="e")
        self.expanding_training_size = tk.Text(self.input_frame)
        self.expanding_training_size.insert(tk.END, '20')
        self.expanding_training_size_label = tk.Label(self.input_frame, text="Training Size:", bg=BACKGROUND_COLOR,
                                                      fg=FOREGROUND_COLOR, anchor="e")
        self.expanding_validation_size = tk.Text(self.input_frame)
        self.expanding_validation_size.insert(tk.END, '0')
        self.expanding_validation_size_label = tk.Label(self.input_frame, text="Validation Size:", bg=BACKGROUND_COLOR,
                                                        fg=FOREGROUND_COLOR, anchor="e")
        self.expanding_gap_size = tk.Text(self.input_frame)
        self.expanding_gap_size.insert(tk.END, '0')
        self.expanding_gap_size_label = tk.Label(self.input_frame, text="Gap Size:", bg=BACKGROUND_COLOR,
                                                 fg=FOREGROUND_COLOR, anchor="e")
        self.expanding_expansion_rate = tk.Text(self.input_frame)
        self.expanding_expansion_rate.insert(tk.END, '1')
        self.expanding_expansion_rate_label = tk.Label(self.input_frame, text="Expansion Rate:", bg=BACKGROUND_COLOR,
                                                       fg=FOREGROUND_COLOR, anchor="e")
        self.normalization_selection.set(self.normalization_options[0])
        self.normalization_selector = tk.OptionMenu(
            self.input_frame,
            self.normalization_selection,
            *self.normalization_options
        )
        self.normalization_selector_label = tk.Label(self.input_frame, text="Normalization Type:",
                                                     bg=BACKGROUND_COLOR, fg=FOREGROUND_COLOR, anchor="e")
        self.training_features = tk.Listbox(self.input_frame, exportselection=0)
        self.training_features_label = tk.Label(self.input_frame, text="Training Features:", bg=BACKGROUND_COLOR,
                                                fg=FOREGROUND_COLOR, anchor="e")
        self.output_features = tk.Listbox(self.input_frame, exportselection=0)
        self.output_features_label = tk.Label(self.input_frame, text="Output Features:", bg=BACKGROUND_COLOR,
                                              fg=FOREGROUND_COLOR, anchor="e")
        self.input_width = tk.Text(self.input_frame)
        self.input_width.insert(tk.END, '3')
        self.input_width_label = tk.Label(self.input_frame, text="Input Width:", bg=BACKGROUND_COLOR,
                                          fg=FOREGROUND_COLOR, anchor="e")
        self.output_width = tk.Text(self.input_frame)
        self.output_width.insert(tk.END, '1')
        self.output_width_label = tk.Label(self.input_frame, text="Output Width:", bg=BACKGROUND_COLOR,
                                           fg=FOREGROUND_COLOR, anchor="e")
        self.stride = tk.Text(self.input_frame)
        self.stride.insert(tk.END, '1')
        self.stride_label = tk.Label(self.input_frame, text="Stride Size:", bg=BACKGROUND_COLOR,
                                     fg=FOREGROUND_COLOR, anchor="e")
        self.gap_size = tk.Text(self.input_frame)
        self.gap_size.insert(tk.END, '0')
        self.gap_size_label = tk.Label(self.input_frame, text="Gap Size:", bg=BACKGROUND_COLOR,
                                       fg=FOREGROUND_COLOR, anchor="e")
        self.epoch = tk.Text(self.input_frame)
        self.epoch.insert(tk.END, '30')
        self.epoch_label = tk.Label(self.input_frame, text="Epoch:", bg=BACKGROUND_COLOR,
                                    fg=FOREGROUND_COLOR, anchor="e")
        self.learning_rate = tk.Text(self.input_frame)
        self.learning_rate.insert(tk.END, '0.0001')
        self.learning_rate_label = tk.Label(self.input_frame, text="Learning Rate:", bg=BACKGROUND_COLOR,
                                            fg=FOREGROUND_COLOR, anchor="e")
        self.train_model_button = tk.Button(
            self.input_frame,
            text="Train Model",
            bg=BUTTON_BACKGROUND,
            fg=BUTTON_FOREGROUND,
            command=lambda: self.train_model()
        )
        self.output_text = tk.Label(self.output_frame)
        self.output_text_label = tk.Label(self.output_frame, text="Output:", bg=BACKGROUND_COLOR, fg=FOREGROUND_COLOR)
        self.save_model_button = tk.Button(
            self.input_frame,
            text="Save Model",
            bg=BUTTON_BACKGROUND,
            fg=BUTTON_FOREGROUND,
            command=lambda: self.save_model()
        )

    def draw(self):
        Menu.draw(self)
        self.input_frame.place(relx=0, rely=0, relwidth=1, relheight=.8)
        self.output_frame.place(relx=0, rely=.8, relwidth=1, relheight=.2)
        self.csv_selector.place(x=_ALIGN_X + 110, y=_ALIGN_Y + 5, width=150)
        self.schema_selector.place(x=_ALIGN_X + 440, y=_ALIGN_Y + 5, width=150)
        self.inputer_selector_label.place(x=_ALIGN_X + 5, y=_ALIGN_Y + 45, width=100)
        self.inputer_selector.place(x=_ALIGN_X + 110, y=_ALIGN_Y + 45, width=100)
        self.split_type_selector_label.place(x=_ALIGN_X + 335, y=_ALIGN_Y + 45, width=100)
        self.split_type_selector.place(x=_ALIGN_X + 440, y=_ALIGN_Y + 45, width=150)
        self.normalization_selector_label.place(x=_ALIGN_X + 655, y=_ALIGN_Y + 45, width=110)
        self.normalization_selector.place(x=_ALIGN_X + 770, y=_ALIGN_Y + 45, width=150)
        self.training_features_label.place(x=_ALIGN_X + 5, y=_ALIGN_Y + 165, width=100)
        self.training_features.place(x=_ALIGN_X + 110, y=_ALIGN_Y + 165, height=100, width=200)
        self.output_features_label.place(x=_ALIGN_X + 335, y=_ALIGN_Y + 165, width=100)
        self.output_features.place(x=_ALIGN_X + 440, y=_ALIGN_Y + 165, height=100, width=200)
        self.epoch_label.place(x=_ALIGN_X + 665, y=_ALIGN_Y + 165, width=100)
        self.epoch.place(x=_ALIGN_X + 770, y=_ALIGN_Y + 165, height=25, width=200)
        self.learning_rate_label.place(x=_ALIGN_X + 665, y=_ALIGN_Y + 205, width=100)
        self.learning_rate.place(x=_ALIGN_X + 770, y=_ALIGN_Y + 205, height=25, width=200)
        self.input_width_label.place(x=_ALIGN_X + 5, y=_ALIGN_Y + 280, width=100)
        self.input_width.place(x=_ALIGN_X + 110, y=_ALIGN_Y + 280, height=25, width=200)
        self.output_width_label.place(x=_ALIGN_X + 335, y=_ALIGN_Y + 280, width=100)
        self.output_width.place(x=_ALIGN_X + 440, y=_ALIGN_Y + 280, height=25, width=200)
        self.stride_label.place(x=_ALIGN_X + 5, y=_ALIGN_Y + 320, width=100)
        self.stride.place(x=_ALIGN_X + 110, y=_ALIGN_Y + 320, height=25, width=200)
        self.gap_size_label.place(x=_ALIGN_X + 335, y=_ALIGN_Y + 320, width=100)
        self.gap_size.place(x=_ALIGN_X + 440, y=_ALIGN_Y + 320, height=25, width=200)
        self.train_model_button.place(x=_ALIGN_X + 110, y=_ALIGN_Y + 360, width=150)
        self.save_model_button.place(x=_ALIGN_X + 440, y=_ALIGN_Y + 360, width=150)
        self.output_text_label.place(x=_ALIGN_X + 5)
        self.output_text.place(x=_ALIGN_X + 10, relx=.05, rely=.05, relwidth=.9, relheight=.9)
        self.draw_split_type_inputs(self.split_type_options[0])

    def hide(self):
        Menu.hide(self)

    def draw_split_type_inputs(self, selection):
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
        self.expanding_expansion_rate_label.place_forget()
        if selection == self.split_type_options[0]:
            self.straight_training_slider_label.place(x=_ALIGN_X + 5, y=_ALIGN_Y + 85, width=100)
            self.straight_training_slider.place(x=_ALIGN_X + 110, y=_ALIGN_Y + 85, width=200, height=40)
            self.straight_validation_slider_label.place(x=_ALIGN_X + 335, y=_ALIGN_Y + 85, width=100)
            self.straight_validation_slider.place(x=_ALIGN_X + 440, y=_ALIGN_Y + 85, width=200, height=40)
        if selection == self.split_type_options[1]:
            self.rolling_training_size_label.place(x=_ALIGN_X + 5, y=_ALIGN_Y + 85, width=100)
            self.rolling_training_size.place(x=_ALIGN_X + 110, y=_ALIGN_Y + 85, width=200, height=25)
            self.rolling_validation_size_label.place(x=_ALIGN_X + 335, y=_ALIGN_Y + 85, width=100)
            self.rolling_validation_size.place(x=_ALIGN_X + 440, y=_ALIGN_Y + 85, width=200, height=25)
            self.rolling_testing_size_label.place(x=_ALIGN_X + 5, y=_ALIGN_Y + 125, width=100)
            self.rolling_testing_size.place(x=_ALIGN_X + 110, y=_ALIGN_Y + 125, width=200, height=25)
            self.rolling_gap_size_label.place(x=_ALIGN_X + 335, y=_ALIGN_Y + 125, width=100)
            self.rolling_gap_size.place(x=_ALIGN_X + 440, y=_ALIGN_Y + 125, width=200, height=25)
            self.rolling_stride_size_label.place(x=_ALIGN_X + 665, y=_ALIGN_Y + 125, width=100)
            self.rolling_stride_size.place(x=_ALIGN_X + 770, y=_ALIGN_Y + 125, width=200, height=25)
        if selection == self.split_type_options[2]:
            self.expanding_training_size_label.place(x=_ALIGN_X + 5, y=_ALIGN_Y + 85, width=100)
            self.expanding_training_size.place(x=_ALIGN_X + 110, y=_ALIGN_Y + 85, width=200, height=25)
            self.expanding_validation_size_label.place(x=_ALIGN_X + 335, y=_ALIGN_Y + 85, width=100)
            self.expanding_validation_size.place(x=_ALIGN_X + 440, y=_ALIGN_Y + 85, width=200, height=25)
            self.expanding_gap_size_label.place(x=_ALIGN_X + 5, y=_ALIGN_Y + 125, width=100)
            self.expanding_gap_size.place(x=_ALIGN_X + 110, y=_ALIGN_Y + 125, width=200, height=25)
            self.expanding_expansion_rate_label.place(x=_ALIGN_X + 335, y=_ALIGN_Y + 125, width=100)
            self.expanding_expansion_rate.place(x=_ALIGN_X + 440, y=_ALIGN_Y + 125, width=200, height=25)

    def save_model(self):
        training_model_file_types = [('All types(*.*)', '*.*')]
        file = asksaveasfile(filetypes=training_model_file_types, defaultextention=training_model_file_types)

    def train_model(self):
        pass

    def generate_training_and_output_features(self):
        self.training_features.delete(0, tk.END)
        self.output_features.delete(0, tk.END)
        for i in self.training_csv:
            self.training_features.insert(tk.END, self.training_csv[i].name)
        for i in self.training_csv:
            self.output_features.insert(tk.END, self.training_csv[i].name)

    def upload_csv(self):
        new_data = filedialog.askopenfile(mode='r', filetypes=[(CSV_FILE_LABEL, CSV_FILE_TYPE)])
        if new_data is not None:
            self.training_csv = pd.read_csv(new_data)
            self.generate_training_and_output_features()

    def upload_schema(self):
        new_data = filedialog.askopenfile(mode='r', filetypes=[(CSV_FILE_LABEL, CSV_FILE_TYPE)])
        if new_data is not None:
            self.training_schema = pd.read_csv(new_data)


class ClimateChangeMenu(Menu):
    def __init__(self, app_frame: tk.Frame):
        Menu.__init__(self, app_frame)
        self.dates = []
        self.generate_plot_button = None  # Type: tk.Button
        self.data_source_table_button = None  # Type: tk.Button
        self.upload_new_data = None  # Type: tk.Button
        self.graph_frame = None  # Type: tk.Frame
        self.tool_frame = None  # Type: tk.Frame
        self.data_selection_box = None  # Type: tk.Listbox
        self.source_data = None  # Type: pd.DataFrame

    def init_ui(self):
        Menu.init_ui(self)
        self.source_data = ClimateAccess.get_source_data()
        self.graph_frame = tk.Frame(master=self.body, bg=BACKGROUND_COLOR)
        self.tool_frame = tk.Frame(master=self.body, bg=BACKGROUND_COLOR)
        self.upload_new_data = tk.Button(
            self.tool_frame,
            text="Upload New Data",
            bg=BUTTON_BACKGROUND,
            fg=BUTTON_FOREGROUND,
            command=lambda: self.upload_data()
        )
        self.generate_plot_button = tk.Button(
            self.tool_frame,
            text="Generate Line Graph",
            bg=BUTTON_BACKGROUND,
            fg=BUTTON_FOREGROUND,
            command=lambda: self.generate_plot()
        )
        self.data_source_table_button = tk.Button(
            self.tool_frame,
            text="Show Source Data Explorer",
            bg=BUTTON_BACKGROUND,
            fg=BUTTON_FOREGROUND,
            command=lambda: self.generate_table()  # Todo: add functionality for this button
        )
        self.data_selection_box = tk.Listbox(
            self.tool_frame,
            bg=BUTTON_BACKGROUND,
            fg=BUTTON_FOREGROUND,
        )

    def draw(self):
        Menu.draw(self)
        self.graph_frame.place(relx=.2, rely=0, relwidth=0.8, relheight=1)
        self.tool_frame.place(relx=0, rely=0, relwidth=.2, relheight=1)
        self.generate_plot_button.place(x=_ALIGN_X + 5, y=_ALIGN_Y + 5, relwidth=.9)
        self.data_source_table_button.place(x=_ALIGN_X + 5, y=_ALIGN_Y + 60, relwidth=.9)
        self.data_selection_box.place(x=_ALIGN_X + 5, y=_ALIGN_Y + 125, relwidth=.9, relheight=.6)
        self.upload_new_data.place(x=_ALIGN_X + 5, rely=.9, relwidth=.9)
        self.populate_data_picker()

    def hide(self):
        Menu.hide(self)

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
        plot1.set_ylabel(str(self.data_selection_box.get(tk.ACTIVE)))
        plot1.set_xlabel('Dates')
        plot1.plot(self.dates, self.source_data[str(self.data_selection_box.get(tk.ACTIVE))])
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
        new_data = filedialog.askopenfile(mode='r', filetypes=[(CSV_FILE_LABEL, CSV_FILE_TYPE)])
        if new_data is not None:
            self.source_data = pd.read_csv(new_data, parse_dates=True)
            self.populate_data_picker()


class AppWindow:
    WINDOW_WIDTH = 1000
    WINDOW_HEIGHT = 600

    _WINDOW_DIMENSIONS = {
        "relx": 0,
        "rely": 0,
        "relwidth": 1.0,
        "relheight": 1.0
    }

    current_width = WINDOW_WIDTH
    current_height = WINDOW_HEIGHT

    menu_list: Dict = {}
    current_menu = None

    def __init__(self, root: tk.Tk):
        self.window = tk.Canvas(root, width=self.WINDOW_WIDTH, height=self.WINDOW_HEIGHT, highlightthickness=0)
        self.frame = tk.Frame(root, bg=BACKGROUND_COLOR)
        self.window.pack(fill='both', expand=True)
        self.frame.place(**self._WINDOW_DIMENSIONS)

        self.window.bind("<Configure>", self.on_resize)

    @staticmethod
    def on_resize(event):
        AppWindow.current_width, AppWindow.current_height = event.width, event.height
        if AppWindow.current_menu is not None:
            AppWindow.current_menu.draw()

    @staticmethod
    def register_menu(screen: Menu, name: Menus):
        AppWindow.menu_list[name] = screen

    @staticmethod
    def display_screen(screen: Menus):
        utils.log(__name__).debug("Opening " + str(screen) + "!")
        if AppWindow.current_menu is not None:
            AppWindow.current_menu.hide()
        AppWindow.current_menu = AppWindow.menu_list[screen]
        AppWindow.current_menu.init_ui()
        AppWindow.current_menu.draw()
