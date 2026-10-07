import unittest


class PackageImportTests(unittest.TestCase):
    def test_dicom_loader_importable(self):
        from imagex.preprocessing.dicom_loader import load_dicom_file

        self.assertTrue(callable(load_dicom_file))
