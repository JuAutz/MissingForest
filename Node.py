import numpy as np


class Node:
    def __init__(self, depth, max_depth: int = None, cores: int = None, criterion: str = "EXHAUSTIVE"):
        self.max_depth = max_depth
        self.depth = depth
        self.splitting_crit = criterion
        self.cores = cores
        self.splitting_field = None
        self.splitting_value = None
        self.class_dict = {}
        self.left_child = None
        self.right_child = None

    def fit(self, input_array: np.ndarray, targets: np.ndarray):
        """ Fit a dataset to this node or one of its children. """

        if self._stopping_criterion_reached(targets):
            self._save_to_dict(targets)
        else:
            self._split(input_array, targets)

    def predict(self, single_input: np.array)->dict:
        """Predict the class of a single input"""
        if self._has_children():
            child = self._matching_child(single_input)
            return child.predict(single_input)
        else:
            return self.class_dict

    def _has_children(self) -> bool:
        if self.left_child or self.right_child:
            return True
        else:
            return False

    def _matching_child(self, single_input):
        if single_input[self.splitting_field] < self.splitting_value:
            return self.left_child
        else:
            return self.right_child

    def _stopping_criterion_reached(self, targets:np.array) -> bool:
        if self.splitting_crit == "GAIN":
            pass  # Todo: Implement
        if self.splitting_crit == "EXHAUSTIVE":
            #Stop iff there is a single class left
            return len(np.unique(targets))==1

    def _save_to_dict(self, targets):
        unique, counts = np.unique(targets, return_counts=True)
        counts = counts / counts.max()
        self.class_dict = dict(zip(unique, counts))

    def _split(self, input_array, targets):

        self._select_split_value(input_array)

        left_mask = input_array[:, self.splitting_field] < self.splitting_value
        left_input = input_array[left_mask]
        left_targets = targets[left_mask]
        self.left_child = Node(self.depth + 1, self.max_depth, self.cores, self.splitting_crit)
        self.left_child.fit(left_input, left_targets)

        right_mask = input_array[:, self.splitting_field] >= self.splitting_value
        right_input = input_array[right_mask]
        right_targets = targets[right_mask]
        self.right_child = Node(self.depth + 1, self.max_depth, self.cores, self.splitting_crit)
        self.right_child.fit(right_input, right_targets)

    def _select_split_value(self, input_array):
        while True:  # Todo. Make more efficient, able to fail
            selected_field = int(np.random.rand() * input_array.shape[1])
            if np.isnan(input_array[:, selected_field]).any():
                continue
            else:
                self.splitting_value = np.median(input_array[:, selected_field])
                self.splitting_field = selected_field
                break
