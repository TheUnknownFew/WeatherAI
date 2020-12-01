import time
import tkinter as tk
from enum import Enum
from threading import Thread
from typing import List, Dict

from AIForecast import utils
from AIForecast.RNN.WeatherForecasting import ForecastingNetwork
from AIForecast.access import WeatherAccess

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


class Menus(Enum):
    """
    Menus is an enum containing named menus that exist in the application.

    Author: Alexander Cherry
    """

    MAIN_MENU = 0
    TEST_MENU = 1
    TRAIN_MENU = 2


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

    def draw(self):
        self.nav_frame.place(x=0, y=0, relwidth=1, height=self.NAV_HEIGHT)
        i = 0
        for button in self.nav_buttons:
            button.place(x=10 + 100*i + 5*i, rely=0.2, relheight=0.55, width=100)
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
        self.target_city_var.set(WeatherAccess.get_cities()[0])

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
        self.output_window.place(x=_ALIGN_X, y=_ALIGN_Y + label_height + 50, relwidth=0.5, relheight=0.5)

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
    Jason, Swatt, and Tiger
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
        super().__init__(app_frame, "Press the button below to generate a prediction:")
        self.radio_group = tk.IntVar()
        self.source_current_radio: tk.Radiobutton = None
        self.source_manual_radio: tk.Radiobutton = None
        self.source_radio_label: tk.Label = None
        self.button_enter: tk.Button = None

    def init_ui(self):
        super().init_ui()
        self.source_current_radio = tk.Radiobutton(
            self.left_pane,
            text="Current",
            variable=self.radio_group,
            value=1,
            borderwidth=5,
            bg=BACKGROUND_COLOR,
            fg=FOREGROUND_COLOR,
            selectcolor='gray',
            command=self._on_radio_current
        )
        self.source_manual_radio = tk.Radiobutton(
            self.left_pane,
            text="Manual",
            variable=self.radio_group,
            value=2,
            borderwidth=5,
            bg=BACKGROUND_COLOR,
            fg=FOREGROUND_COLOR,
            selectcolor='gray',
            command=self._on_radio_manual
        )
        self.source_radio_label = tk.Label(self.left_pane, text="temporary", bg=BACKGROUND_COLOR, fg=FOREGROUND_COLOR)
        self.button_enter = tk.Button(
            self.left_pane,
            text="Enter Data",
            bg=BUTTON_BACKGROUND,
            fg=BUTTON_FOREGROUND,
            command=self._on_enter_data
        )
        self.source_current_radio.select()
        self.source_current_radio.invoke()
        self.button_execute.config(command=self._on_execute)

    def draw(self):
        super().draw()
        label_height = self.get_label_height()
        self.source_current_radio.place(x=_ALIGN_X, y=_ALIGN_Y + label_height)
        radio_width = self.source_current_radio.winfo_reqwidth()
        self.source_manual_radio.place(x=_ALIGN_X + radio_width + 5, y=_ALIGN_Y + label_height)
        self.source_radio_label.place(x=_ALIGN_X, y=_ALIGN_Y * 7)
        self.button_enter.place(x=_ALIGN_X + 5, y=_ALIGN_Y * 11)

    def hide(self):
        super().hide()
        self.source_manual_radio.destroy()
        self.source_manual_radio.destroy()
        self.source_radio_label.destroy()
        self.button_enter.destroy()

    def _on_radio_current(self):
        """
        run_radio_1 is the command routine for source_current_radio.
        """
        self.source_radio_label.config(text="Use current data to make a prediction.")
        self.button_execute.config(state='normal')
        self.surrounding_cities_select.config(state='disabled')
        self.target_city_select.config(state='disabled')
        self.button_enter.config(state='disabled')

    def _on_radio_manual(self):
        """
        run_radio_2 is the command routine for source_manual_radio.
        """
        self.source_radio_label.config(text="Manually input data to make a prediction.")
        self.button_execute.config(state='normal')
        self.surrounding_cities_select.config(state='normal')
        self.target_city_select.config(state='normal')
        self.button_enter.config(state='normal')

    def _on_enter_data(self):
        self.output_window.config(text="This feature has not been implemented yet.")

    def _on_execute(self):
        import numpy as np
        model, model_mean, model_std = ForecastingNetwork.get_saved_model()
        current_weather = WeatherAccess.get_current_weather_at('Erie')
        current_weather = ForecastingNetwork.scale(current_weather, model_mean, model_std)
        prediction = model.predict(current_weather)
        prediction = ForecastingNetwork.unscale(prediction, model_mean['temperature'], model_std['temperature'])
        print(prediction)


class TrainMenu(IOMenu):
    def __init__(self, app_frame: tk.Frame):
        super().__init__(app_frame, "Press the button below to begin training:")
        self.radio_group = tk.IntVar()
        self.start_year_var = tk.IntVar()
        self.end_year_var = tk.IntVar()
        self.source_historic_radio: tk.Radiobutton = None
        self.source_user_radio: tk.Radiobutton = None
        self.source_radio_label: tk.Label = None
        self.start_year_label: tk.Label = None
        self.start_year_select: tk.OptionMenu = None
        self.end_year_label: tk.Label = None
        self.end_year_select: tk.OptionMenu = None
        self.future_time_entry_label: tk.Label = None
        self.future_time_entry: tk.Entry = None

    def init_ui(self):
        super().init_ui()
        self.source_historic_radio = tk.Radiobutton(
            self.left_pane,
            text="Historic",
            variable=self.radio_group,
            value=1,
            borderwidth=5,
            bg=BACKGROUND_COLOR,
            fg=FOREGROUND_COLOR,
            selectcolor='gray',
            command=self._on_radio_historic
        )
        self.source_user_radio = tk.Radiobutton(
            self.left_pane,
            text="User",
            variable=self.radio_group,
            value=2,
            borderwidth=5,
            bg=BACKGROUND_COLOR,
            fg=FOREGROUND_COLOR,
            selectcolor='gray',
            command=self._on_radio_user
        )
        self.source_radio_label = tk.Label(self.left_pane, text="temporary", bg=BACKGROUND_COLOR, fg=FOREGROUND_COLOR)
        self.start_year_label = tk.Label(
            self.left_pane,
            text="Select a starting year:",
            bg=BACKGROUND_COLOR,
            fg=FOREGROUND_COLOR
        )
        self.start_year_select = tk.OptionMenu(self.body, self.start_year_var, *WeatherAccess.years)
        self.end_year_label = tk.Label(
            self.left_pane,
            text="Select an ending year:",
            bg=BACKGROUND_COLOR,
            fg=FOREGROUND_COLOR
        )
        self.end_year_select = tk.OptionMenu(self.body, self.end_year_var, *WeatherAccess.years)
        self.future_time_entry_label = tk.Label(
            self.left_pane,
            text="Train to hours in future:",
            bg=BACKGROUND_COLOR,
            fg=FOREGROUND_COLOR
        )
        self.future_time_entry = tk.Entry(self.body)
        self.source_historic_radio.select()
        self.source_historic_radio.invoke()
        self.button_execute.config(command=self._on_execute)

    def draw(self):
        super().draw()
        label_height = self.get_label_height()
        self.source_historic_radio.place(x=_ALIGN_X, y=_ALIGN_Y + label_height)
        radio_width = self.source_historic_radio.winfo_reqwidth()
        self.source_user_radio.place(x=_ALIGN_X + radio_width + 5, y=_ALIGN_Y + label_height)
        self.source_radio_label.place(x=_ALIGN_X, y=_ALIGN_Y * 7)
        self.start_year_label.place(x=_ALIGN_X, y=_ALIGN_Y * 7 + label_height + 5)
        year_label_width = self.start_year_label.winfo_reqwidth()
        self.end_year_label.place(x=_ALIGN_X + year_label_width + 5, y=_ALIGN_Y * 7 + label_height + 5)
        self.start_year_select.place(x=_ALIGN_X, y=_ALIGN_Y * 10 + label_height + 5)
        self.end_year_select.place(x=_ALIGN_X + year_label_width + 5, y=_ALIGN_Y * 10 + label_height + 5)
        year_select_height = self.end_year_select.winfo_reqheight()
        self.future_time_entry_label.place(x=_ALIGN_X, y=_ALIGN_Y * 15 + year_select_height + 5)
        self.future_time_entry.place(
            x=_ALIGN_X + self.future_time_entry_label.winfo_reqwidth() + 5,
            y=_ALIGN_Y * 15 + year_select_height + 5,
            width=50
        )

    def hide(self):
        super().hide()
        self.source_historic_radio.destroy()
        self.source_user_radio.destroy()
        self.source_radio_label.destroy()
        self.start_year_label.destroy()
        self.start_year_select.destroy()
        self.end_year_label.destroy()
        self.end_year_select.destroy()
        self.future_time_entry_label.destroy()
        self.future_time_entry.destroy()

    def _on_radio_historic(self):
        self.source_radio_label.config(text="This option uses historic data to train the model.")
        self.button_execute.config(state="normal")
        self.surrounding_cities_select.config(state="normal")
        self.target_city_select.config(state="normal")
        self.start_year_var.set(WeatherAccess.get_years()[0])
        self.end_year_var.set(WeatherAccess.get_years()[1])

    def _on_radio_user(self):
        self.source_radio_label.config(text="this feature is currently unavailable.")
        self.surrounding_cities_select.config(state='disabled')
        self.target_city_select.config(state='disabled')

    def _on_execute(self):
        self.output("Please wait while the network trains!")
        data = WeatherAccess.query_historical_data(self.selected_cities, self.start_year, self.end_year)
        rnn = ForecastingNetwork(data)
        rnn.train_network(self.hours)
        print(rnn.get_example_predictions())
        self.output(
            """The network has finished training!
            You may now go to the Test menu to make predictions.
            --------
            Start Year: """ + str(self.start_year) + """
            End Year: """ + str(self.end_year) + """
            Hours into the Future: """ + str(self.hours) + """
            Example prediction for """ + self.target_city_var.get()
        )

    @property
    def start_year(self):
        return self.start_year_var.get()

    @property
    def end_year(self):
        return self.end_year_var.get()

    @property
    def hours(self):
        return self.future_time_entry.get()


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
