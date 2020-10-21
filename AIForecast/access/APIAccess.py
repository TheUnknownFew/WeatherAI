from pyowm.weatherapi25.location import Location
from AIForecast.utils import owm_access as owm


def get_weather_data_for(city: str = '', city_state: str = ''):
    """
    The city parameters refers to the city name.
    The city_state parameter refers to the state or country that city is located in.
    Returns the weather data for a city in a country.
    Raises ValueError if city could not be found.
    """
    weather_loc = owm.city_id_registry().locations_for(city_name=city, country=city_state)
    print(len(weather_loc))


if __name__ == '__main__':
    get_weather_data_for('Erie', 'PA')
