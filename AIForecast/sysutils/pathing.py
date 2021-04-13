import os
import shutil
from enum import Enum, unique
from os import path as filesys
from os.path import join as mkpath
from typing import List, Union

from AIForecast.sysutils.sysexceptions import ModificationError

DEFAULT_ROOT_NAME = 'AIClimateChange'


@unique
class FolderStructure(Enum):
    """
    | Contains a list of directory locations maintained by the system.
    | Specifically, the FolderStructure Enum lists out the root directory and sub-directories maintained by the system.
    | Use this Enum to get paths to specific containing directories.
    |
    | **Developer Notes:**
    | When adding new enumerations to FolderStructure, order matters. Enumerations in FolderStructure are initialized in
      the order they appear, thus when defining the 'parent' parameter, the parent enumeration must already be defined
      and initialized. Additionally, when defining the 'parent' parameter, provide a string with the name of the
      enumeration that is the parent. An empty string as the parent can be used to represent a base directory.
    """

    ROOT_DIR = ('', mkpath('..', '..', DEFAULT_ROOT_NAME))
    """
    | Location: ./AIClimateChange
    | Path:     ../../AIClimateChange 
    """
    CLIMATE_DATA_DIR = ('ROOT_DIR', mkpath('data', 'modeling'))
    """
    | Location: ./data/modeling
    | Path:     ../../AIClimateChange/data/modeling
    """
    WEATHER_DATA_DIR = ('ROOT_DIR', mkpath('data', 'weather'))
    """
    | Location: ./data/weather
    | Path:     ../../AIClimateChange/data/weather
    """
    LOGS_DIR = ('ROOT_DIR', mkpath('logs'))
    """
    | Location: ./logs
    | Path:     ../../AIClimateChange/logs
    """
    MODEL_SCHEMA_DIR = ('ROOT_DIR', mkpath('models', 'schema'))
    """
    | Location: ./models/schema
    | Path:     ../../AIClimateChange/models/schema
    """
    TRAINED_MODELS_DIR = ('ROOT_DIR', mkpath('models', 'trained'))
    """
    | Location: ./models/trained
    | Path:     ../../AIClimateChange/models/trained
    """

    def __init__(self, parent, directory):
        if parent != '' and parent not in self._member_map_:
            raise AttributeError(f'Enumeration {parent} has either not been initialized yet, or does not exist.')
        self.__parent = parent if parent == '' else self._member_map_[parent].get_path()
        self.__location = directory

    def set_path(self, location):
        """
        Sets the path to the ROOT directory.
        :param location: The new location of the root directory.
        :raises ModificationError: raised if an operation attempts to modify the path of a directory other than the ROOT
        directory.
        """
        if self.name != 'ROOT_DIR':
            raise ModificationError(f'{self.name} is a sub-directory of the root directory and should not be modified.')
        self.__location = location

    def get_path(self):
        """
        :return: Returns the absolute path the directory location.
        """
        return filesys.abspath(mkpath(self.__parent, self.__location))


FStruct = FolderStructure   # Alias assignment for FolderStructure.


@unique
class SysFiles(Enum):
    """
    Contains a list of files that are actively maintained by the system. These are files the system needs to access
    in order to be operable. Other files such as model schema files are not listed here since the system does not
    depend on their existence to operate.
    """
    OWN_KEY_FILE = ('owm_apikey', 'txt', FStruct.ROOT_DIR.get_path())
    """
    """
    SAVE_STATE_FILE = ('save_state', 'json', FStruct.ROOT_DIR.get_path())
    """
    """

    def __init__(self, filename, ext, path_to):
        self.__name = filename
        self.__ext = ext
        self.__filename = f'{filename}.{ext}'
        self.__containing_dir = path_to
        self.__path = mkpath(path_to, self.__filename)

    def get_file_extension(self) -> str:
        """
        :return: Returns the file extension.
        """
        return self.__ext

    def get_file_name(self) -> str:
        """
        :return: Returns the file name.
        """
        return self.__name

    def get_file(self) -> str:
        """
        :return: Returns the file name with its extension.
        """
        return self.__filename

    def get_containing_dir(self):
        """
        :return: Returns the absolute path to the containing directory.
        """
        return self.__containing_dir

    def get_path(self):
        """
        :return: Returns the absolute path to the file.
        """
        return self.__path


def make_root_directory(root_dir: str = None):
    """
    | Constructs the system's working root environment if the root environment has been changed, or if the root
      environment does not exist.
    |
    | **Different cases:**
    | - if **root_dir** is None, then the root environment is created at the default location.
    | - if the root directory is being changed and a current root already exists, the location is changed and the old
      file tree is copied over.
    | - if the root directory is not change, then the file structure of the current root directory is ensured to match
    | with the defined directory tree structure defined in the FolderStructure Enum.
    |
    | **Notes:**
    | If a new location for the root directory is being set, the new directory should be empty. Otherwise, the
      directory structure will be made within an additional sub-directory contained in **root_dir**.
    :param root_dir: A path to the new working root directory.
    :raises NotADirectoryError: raised if root_dir does not lead to an existing directory location or is a file.
    """
    root = FStruct.ROOT_DIR.get_path()
    location = root if root_dir is None else filesys.abspath(root_dir)
    location_exists = filesys.exists(location)
    if root_dir is None and not location_exists:
        os.mkdir(location)
    if not location_exists or filesys.isfile(location):
        raise NotADirectoryError(f"'{location}' - path is not a directory. ")
    if location != root:
        location = location if len(os.listdir(location)) == 0 else mkpath(location, DEFAULT_ROOT_NAME)
        if filesys.exists(root):
            shutil.copytree(root, location, dirs_exist_ok=True)
        FStruct.ROOT_DIR.set_path(location)
    for dir_path in FStruct.__members__.values():
        os.makedirs(dir_path.get_path(), exist_ok=True)


def get_files(directory_location: FolderStructure, file_names: Union[List[str], str] = None, match_all: bool = True):
    """
    | Helper function used to get the paths of files contained within directories of the system.
    :param directory_location: A location within the system's root directory structure.
    :param file_names:
        default = None<br/>
        None - retrieves a list of file paths within the specified directory location.<br/>
        List of file names - retrieves a list of file paths that match the file names listed. Check match_all for more
        details on matching criteria. Names must include file extension.<br/>
        string - retrieves a single path to a file with the given name. Name must include file extension.<br/>
    :param match_all:
            default = True<br/>
            True - if a list of file names is given, all files must match to return a list of paths.<br/>
            False - if a list of file names is given, only file names that have a match will have their paths return.
    :return: Returns:<br/>
            - If file_names is None, a list of all files in the directory location are returned.<br/>
            - Else, one or more paths to files in the directory location are returned.<br/>
            - If no files were found, an empty list is returned.
    """
    location = directory_location.get_path()
    all_files = [f_name for f_name in os.listdir(location) if filesys.isfile(mkpath(location, f_name))]
    if file_names is None:
        return [mkpath(location, file) for file in all_files]
    if isinstance(file_names, str) and file_names in all_files:
        file = [x for x in all_files if x == file_names]
        return mkpath(location, file[0]) if len(file) > 0 else []
    match, existing_files = all if match_all else any, [(file in all_files) for file in file_names]
    if isinstance(file_names, List) and match(existing_files):
        return [mkpath(location, file) for file, exists in zip(file_names, existing_files) if exists]


if __name__ == '__main__':
    print('')
