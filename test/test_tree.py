import pytest
import numpy as np
from Tree import Tree


@pytest.fixture(scope='module')
def base_tree():
    return Tree()


@pytest.fixture(scope='module')
def simple_single_data():
    return np.array([[1, 3, 5]]), np.array(["A"])


class TestSingleTree():
    def test_single_data(self, base_tree, simple_single_data):
        base_tree.fit(simple_single_data[0], simple_single_data[1])
        assert base_tree.predict(simple_single_data[0])==simple_single_data[1]

