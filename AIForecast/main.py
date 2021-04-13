# This is the main file
from AIForecast import sysutils
import tkinter as tk

from AIForecast.weather import dataaccess
from AIForecast.ui.widgets import AppWindow, Menus, MainMenu, TestMenu, TrainMenu


def main():
    sysutils.log(__name__).debug('Starting AI-Weather Forecast!')

    dataaccess.load_historical_data()

    root = tk.Tk()
    root.title("AI-Weather Forecast")
    root.minsize(AppWindow.WINDOW_WIDTH, AppWindow.WINDOW_HEIGHT)

    app = AppWindow(root)
    app.register_menu(MainMenu(app.frame), Menus.MAIN_MENU)
    app.register_menu(TestMenu(app.frame), Menus.TEST_MENU)
    app.register_menu(TrainMenu(app.frame), Menus.TRAIN_MENU)
    app.display_screen(Menus.MAIN_MENU)

    root.mainloop()


if __name__ == '__main__':
    main()
