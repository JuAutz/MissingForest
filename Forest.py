import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# Subset modes
ALL = "ALL"
RANDOM = "RANDOM"

# Variants
ADHOC = "ADHOC"
BOOSTED = "BOOSTED"

# Tokens
NAN = "NaN"
from logging import Logger

log = Logger(__file__)


class Forest:
    def __init__(self, max_depth=None, cores=None, variant: str = ADHOC):
        self.max_depth = max_depth
        self.cores = cores
        self.trees = {}
        self.variant = variant
        self.input_data = None
        self.targets = None
        self.saved_trees = {}
        if self.variant == BOOSTED:
            self.svm = SVC()
            self.encoder = LabelEncoder()

    def fit(self, input_array: np.ndarray, targets: np.array):
        if self.variant == ADHOC:
            self._fit_ADHOC(input_array, targets)
        elif self.variant == BOOSTED:
            self._fit_BOOSTED(input_array, targets)
        else:
            pass

    def predict(self, input_array: np.ndarray):
        if self.variant == ADHOC:
            return self._predict_ADHOC(input_array)
        elif self.variant == BOOSTED:
            return self._predict_BOOSTED(input_array)

    def _fit_ADHOC(self, input_array, targets):
        if self.input_data is None and self.targets is None:
            self.input_data = input_array
            self.targets = targets
        else:
            # Todo: Should additional data be added online? No issue to implement, makes comparison iffy
            raise RuntimeError("Forest already fed with data")

    def _predict_ADHOC(self, input_array):
        # Select matching subset of data
        results = []

        for input in input_array:  # Todo: Iterate over patterns instead of inputs

            pattern = np.isnan(input)

            # Select the part of the saved data that matches the pattern of missing values of the input
            matching_input = self.input_data[np.where((np.isnan(self.input_data) == pattern).all(axis=1))]
            matching_targets = self.targets[np.where((np.isnan(self.input_data) == pattern).all(axis=1))]
            forest = RandomForestClassifier()  # Todo: Parameters??
            no_nan_matching_input = matching_input[:, ~pattern]
            if no_nan_matching_input.shape[1] > 0:
                forest.fit(no_nan_matching_input, matching_targets)  #

                no_nan_input = input[~pattern]
                result = forest.predict(no_nan_input.reshape(1, -1))[0]  # Due to single sample

                self.saved_trees[tuple(pattern)] = forest  # Todo: Make use of saved trees

            else:  # Choose from the known classes
                result = np.random.choice(matching_targets)

            results.append(result)

        return results

    def _fit_BOOSTED(self, input_array, targets):
        # Resample data
        number_of_unique_patterns = len(np.unique(np.isnan(input_array), axis=1))
        # idea: Number of trees equals to number of patterns=>
        # More complicates data requires more complicated model <==> higher number of trees.
        for i in range(number_of_unique_patterns):  # Todo: Parallize
            # Choose feature subset
            number_of_features = int(np.math.sqrt(input_array.shape[1]))  # Root of features is standard ->wiki link
            selected_features = np.random.choice(input_array.shape[1], number_of_features, replace=False)

            # Draw data where None of the features are nan, reduce features to selected
            sub_feature_array = input_array[:, selected_features]
            mask = np.isnan(sub_feature_array).any(axis=1)
            mask = np.invert(mask, dtype=bool)

            viable_input = input_array[mask][:, selected_features]
            viable_targets = targets[mask]

            subsampled_input = viable_input[np.random.choice(range(len(viable_input)), len(viable_input))]
            subsampled_targets = viable_targets[np.random.choice(range(len(viable_targets)), len(viable_targets))]

            # Generate trees:
            next_tree = DecisionTreeClassifier()
            next_tree.fit(subsampled_input, subsampled_targets)
            self.trees[tuple(selected_features)] = next_tree

            # Generate learner data: (samples,output_of_trees) as x, target class as y
            learner_input = self._evaluate_trees(input_array)
            assert len(learner_input) == len(targets)  # Todo: Check this in detail

            # Flatten for encoding, unflatten for learning
            input_shape = learner_input.shape
            flat_input = learner_input.flatten()
            self.encoder.fit(flat_input)
            encoded_input = self.encoder.transform(flat_input)
            encoded_input = encoded_input.reshape(input_shape)

            # Train learner
            self.svm.fit(encoded_input, targets)

    def _predict_BOOSTED(self, input_array):
        learner_input = self._evaluate_trees(input_array)
        # Flatten for encoding, unflatten for learning
        input_shape = learner_input.shape
        flat_input = learner_input.flatten()

        encoded_input = self.encoder.transform(flat_input)
        encoded_input = encoded_input.reshape(input_shape)

        predictions = self.svm.predict(encoded_input)
        return predictions

    def _evaluate_trees(self, input_array) -> np.ndarray:
        tree_result = []

        for selected_features, tree in self.trees.items():  # Todo: Parralize
            sub_feature_array = input_array[:, selected_features]
            # Filter out Nans,
            nan_mask = np.isnan(sub_feature_array).any(axis=1)
            nan_array = np.where(nan_mask)
            nan_result = np.ones(len(nan_array), dtype=object)
            nan_result.fill(NAN)
            clean_array = np.where(~nan_mask)
            clean_input = sub_feature_array[clean_array]

            # Predict non-Nans
            if clean_input.shape[0] > 0:  # Catch the empty case
                clean_result = tree.predict(clean_input)
            else:
                clean_result = []

            # Merge | Works only for 1D Data
            final_result = np.zeros(len(input_array), dtype=object)
            np.put(final_result, nan_array, nan_result)
            np.put(final_result, clean_array, clean_result)
            tree_result.append(final_result)

        return np.array(tree_result).transpose()

    def __str__(self):
        return "%s-Forest id: %s" % (self.variant, id(self))
