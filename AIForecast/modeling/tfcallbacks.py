from tensorflow import keras

from AIForecast.ui.widgets import OutputWindow


class OutputEpoch(keras.callbacks.Callback):
    def __init__(self, output_window: OutputWindow, total_epochs: int):
        super().__init__()
        self.output_window: OutputWindow = output_window
        self.total_epochs: int = total_epochs
        self.progress_bar_width = 30

    def on_epoch_end(self, epoch, logs=None):
        epoch += 1
        progress = int(self.progress_bar_width * (epoch / self.total_epochs))
        bar = f'[{"=" * progress}{"." * (self.progress_bar_width - progress)}]'
        output = f'Your model is being trained. This may take a while.\n' \
                 f'Epoch {epoch}: {int((epoch / self.total_epochs) * 100)}% {bar}\n'
        for metric in [f' - {key}: {val}\n' for key, val in logs.items()]:
            output += metric
        self.output_window.output(output)


class CancelModelTraining(keras.callbacks.Callback):
    def __init__(self, output_window: OutputWindow):
        super().__init__()
        self.canceled = False
        self.output_window: OutputWindow = output_window

    def on_epoch_end(self, epoch, logs=None):
        if self.canceled:
            self.output_window.output('Training has been canceled!')
            self.model.stop_training = self.canceled

    def cancel_training(self):
        self.canceled = True
