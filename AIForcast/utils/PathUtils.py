import os


class PathUtils:
    # Private:

    # Working Directory of the software.
    # This variable should remain constant.
    _WORKING_DIR = os.getcwd()
    # Name of the data directory.
    # The data directory stores all relevant system files.
    # This variable should remain constant.
    DATA_DIR = 'data'
    # Name of the hdf5 saves directory.
    # This directory is located in the data directory.
    # This variable should remain constant.
    MODEL_DIR = 'models'
    # Name of the pkl saves directory.
    # This directory is located in the data directory.
    # This variable should remain constant.
    PKL_DIR = 'pickles'
    # Optional save path for software files.
    _save_path = None
    # Set to true if a custom save path has been set.
    _is_custom_path = False

    # Public:
    def set_save_path(self, path: str):
        self._save_path = path
        self._is_custom_path = True

    def get_data_path(self):
        return

    def _get_base_path(self):
        return self._save_path \
            if self._is_custom_path \
            else self._WORKING_DIR
