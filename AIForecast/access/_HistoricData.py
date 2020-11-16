from datetime import datetime


class HistoricData:
    def __int__(self):
        self.date: datetime = None
        self.temperature: float = None
        self.min_temperature: float = None
        self.max_temperature: float = None
        self.pressure: float = None
        self.humidity: float = None
        self.wind_speed: float = None
        self.wind_direction: float = None

    def get_data_point(self):
        return [
            self.date.month,
            self.date.day,
            self.date.hour,
            self.min_temperature,
            self.max_temperature,
            self.pressure,
            self.humidity,
            self.wind_speed,
            self.wind_direction,
            self.temperature
        ]
