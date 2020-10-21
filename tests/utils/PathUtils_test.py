from AIForecast.utils import path_utils as pu
from definitions import ROOT_DIR

# Tests:
if __name__ == '__main__':
    assert pu._get_base_path() == ROOT_DIR
    assert pu.get_owm_apikey() == 'aa91a3b5b86f47f610e04485584c5693'
    pu.set_save_path('hello world')
    assert pu._get_base_path() == pu._save_path