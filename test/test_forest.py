import pytest
import numpy as np

from Forest import Forest

@pytest.fixture(scope='module')
def simple_single_data():
    return np.array([[1, 3, 5]]), np.array(["A"])





class TestAdhoc():

    def test_single_data(self, simple_single_data):
        forest=Forest(variant="ADHOC")
        forest.fit(simple_single_data[0], simple_single_data[1])
        assert forest.predict(simple_single_data[0])==simple_single_data[1]
