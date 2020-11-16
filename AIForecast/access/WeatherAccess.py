from datetime import datetime
from typing import List, Set

import pandas as pd
from pyowm.weatherapi25.weather import Weather

from AIForecast import utils
from AIForecast.utils import owm_access as owm, PathUtils, DataUtils

DEG_F = 'fahrenheit'
DEG_C = 'celsius'
DEG_K = 'kelvin'

cvs_columns = [
    'timestamp', 'city_name', 'temperature',
    'temperature_min', 'temperature_max', 'pressure',
    'humidity', 'wind_velocity_x', 'wind_velocity_y',
    'day_x', 'day_y', 'year_x', 'year_y'
]
historic_data: pd.DataFrame = None
cities: Set[str] = None
years: Set[int] = None


def get_years():
    return list(years)


def get_cities():
    return list(cities)


def load_historical_data() -> None:
    """
    Loads in data/Data.json
    """
    global historic_data, cities, years
    utils.log(__name__).debug("Loading historic data!")
    cvs_path = PathUtils.get_file(PathUtils.get_data_path(), "Data.cvs")
    if PathUtils.file_exists(cvs_path):
        utils.log(__name__).debug("Loading historic data from CVS.")
        historic_data = pd.read_csv(cvs_path)
    else:
        utils.log(__name__).debug("No Data.cvs found! Loading and converting Data.json to Data.cvs.")
        import ijson
        json_matrix = []
        with open(PathUtils.get_file(PathUtils.get_data_path(), "Data.json")) as f:
            for item in ijson.items(f, 'item'):
                wind_x, wind_y = DataUtils.vector_2d(item['wind']['speed'], item['wind']['deg'])
                day_x, day_y, year_x, year_y = DataUtils.periodicity(item['dt'])
                json_matrix.append([
                    datetime.fromtimestamp(item['dt']).isoformat(),
                    item['city_name'],
                    item['main']['temp'],
                    item['main']['temp_min'],
                    item['main']['temp_max'],
                    item['main']['pressure'],
                    item['main']['humidity'],
                    wind_x, wind_y,
                    day_x, day_y,
                    year_x, year_y
                ])
        historic_data = pd.DataFrame(json_matrix, columns=cvs_columns)
        utils.log(__name__).debug("Saving data to CVS.")
        historic_data.to_csv(cvs_path)
    cities = {city for city in historic_data['city_name']}
    years = {datetime.fromisoformat(timestamp).year for timestamp in historic_data['timestamp']}
    utils.log(__name__).debug("Finished loading historic weather data.")


def query_historical_data(training_cities: List[str], start_year, end_year) -> pd.DataFrame:
    """

    """
    # https://machinelearningmastery.com/how-to-use-the-timeseriesgenerator-for-time-series-forecasting-in-keras/
    if historic_data is None:
        raise ValueError

    if int(start_year) > int(end_year):
        raise IndexError

    utils.log(__name__).debug("Processing data!")
    # Filter the historical matching the passed parameter criteria.
    return historic_data.loc[((historic_data['timestamp'] >= datetime(int(start_year), 1, 1).isoformat())
                             & (historic_data['timestamp'] < datetime(int(end_year), 1, 1).isoformat()))
                             & historic_data['city_name'].isin(training_cities)][cvs_columns[2:]]


def get_current_weather_at(city: str = '', city_state: str = '') -> Weather:
    """
    The city parameters refers to the city name.
    The city_state parameter refers to the state or country that city is located in.
    Returns the weather data for a city in a country.
    Raises ValueError if city could not be found.
    Raises IndexError if more than one city has been returned.
    """
    weather_loc = owm.city_id_registry().locations_for(city_name=city, country=city_state)
    if len(weather_loc) > 1:
        raise IndexError
    return owm.weather_manager().weather_at_id(weather_loc[0].id).weather


if __name__ == '__main__':
    load_historical_data()
    query_historical_data(['Pittsburgh', 'Buffalo'], 2018, 2019)
