import pytest
import numpy as np
from os.path import join, dirname
import sys

sys.path.append(join("..", "..", dirname(__file__)))

from Forest import Forest


@pytest.fixture(scope='module')
def simple_single_data():
    return np.array([[1, 3, 5]]), np.array(["A"])


@pytest.fixture(scope='module')
def simple_data():
    return np.array([[1, 3, 5], [100, 900, 110]]), np.array(["A", "B"])


@pytest.fixture(scope='module')
def simple_nan_single_data():
    return np.array([[1, np.nan, 5]]), np.array(["A"])

@pytest.fixture(scope='module')
def simple_nan_data():
    return np.array([[1, np.nan, 5], [100, 900, np.nan]]), np.array(["A", "B"])


class TestAdhoc():
    def test_single_data_ADHOC(self, simple_single_data):
        forest = Forest(variant="ADHOC")
        forest.fit(simple_single_data[0], simple_single_data[1])
        assert forest.predict(simple_single_data[0]) == simple_single_data[1]

    def test_single_data_BOOSTED(self, simple_data):
        forest = Forest(variant="BOOSTED")
        forest.fit(simple_data[0], simple_data[1])
        assert (forest.predict(simple_data[0]) == simple_data[1]).all() #Todo: Make deterministic

    def test_simple_nan_data_ADHOC(self, simple_nan_single_data):
        forest = Forest(variant="ADHOC")
        forest.fit(simple_nan_single_data[0], simple_nan_single_data[1])
        assert forest.predict(simple_nan_single_data[0]) == simple_nan_single_data[1]

    def test_simple_nan_data_BOOSTED(self, simple_nan_data):
        forest = Forest(variant="BOOSTED")
        forest.fit(simple_nan_data[0], simple_nan_data[1])
        assert (forest.predict(simple_nan_data[0]) == simple_nan_data[1]).all()
