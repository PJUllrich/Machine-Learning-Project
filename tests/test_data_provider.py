from unittest import TestCase

import config
from data_provider import DataProvider


class TestDataProvider(TestCase):
    def test_get_data(self):
        data = DataProvider.get_data()

        # Assert that data was returned
        assert data is not None

        # Assert that the correct number of painters was returned
        assert len(data) == config.NUM_PAINTERS

        # Assert that all painters have a non-zero list of paintings
        assert all([len(pair[1]) > 0] for pair in data)
