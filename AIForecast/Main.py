# This is the main file
# from tkinter import Tk, StringVar

from AIForecast import utils as logger
from AIForecast.utils import path_utils
# from AIForecast.ui.GUI import Window


def main():
    logger.log(__name__).debug('Starting AI-Weather Forecast!')
    # root = Tk()

    # variable = StringVar()
    # root.geometry("1000x600")
    # app = Window(root)
    # root.mainloop()


if __name__ == '__main__':
    main()
