from AIForecast.sysutils import pathing
from pyowm.owm import OWM
import logging as logger
import os

owm_access = 5  # OWM(pathing.get_owm_apikey())
logger.basicConfig(format=logger.BASIC_FORMAT, level=logger.DEBUG)


# Utility function for logging basic messages.
# 'from AIForecast import sysutils as logger' to call log.
# Todo: Potentially move this to its own class.
def log(name):
    return logger.getLogger(name)
