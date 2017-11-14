import numpy as np

from Node import Node


class Tree:
    """
    Class that represents a singe tree.

     The tree can handle missing data and """

    def __init__(self, max_depth=None, cores=None):
        self.max_depth = max_depth
        self.cores = cores
        self.root = Node(1, max_depth, cores)

    def fit(self, input_array: np.ndarray, target: np.ndarray):
        """Fits the internal model to a dataset.

         :param input_array: Of the shape [set_size,input_dim]
         :param target: The array of the classes, of the shape [input_length]
         """

        self.root.fit(input_array, target)

    def predict(self, single_input: np.ndarray) -> np.array:
        result_dict = self.root.predict(single_input)
        result = max(result_dict,key=lambda key:result_dict[key]) #get highest valued key from dict/ most likely class

        return result
