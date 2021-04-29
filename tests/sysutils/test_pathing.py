import unittest
import shutil
import os

from AIForecast.sysutils import pathing
from AIForecast.sysutils.pathing import FolderStructure


class FileStructureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_location = os.path.join('..', '..', 'AIClimateChange_tmp')

    def test_create_root_directory(self):
        os.mkdir(self.tmp_location)
        self.assertTrue(os.path.exists(FolderStructure.ROOT_DIR.get_path()), 'Root directory d')
        shutil.copytree(FolderStructure.ROOT_DIR.get)

    def test_create_new_root_directory(self):
        os.mkdir(self.tmp_location)
        pathing.make_root_directory(self.tmp_location)
        for root_dir in FolderStructure.__members__.values():
            self.assertTrue(os.path.exists(root_dir.get_path()))
        shutil.rmtree(self.tmp_location)

    def tearDown(self) -> None:
        del self.tmp_location


if __name__ == '__main__':
    unittest.main()
