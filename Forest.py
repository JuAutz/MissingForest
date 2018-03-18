import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import LabelEncoder
from multiprocessing import Pool, Manager
import psutil

# Subset modes
ALL = "ALL"
RANDOM = "RANDOM"

# Variants
ADHOC = "ADHOC"
BOOSTED = "BOOSTED"

# Globals

# Tokens
NAN = "NaN"
from logging import Logger, FileHandler, Formatter, INFO, DEBUG

log = Logger(__file__)
hdlr = FileHandler('.log')
formatter = Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
log.addHandler(hdlr)
log.setLevel(DEBUG)


class Forest:
    def __init__(self, max_depth=None, cores=6, variant: str = ADHOC, number_of_trees=100):
        self.max_depth = max_depth
        self.cores = cores
        self.trees = {}
        self.variant = variant
        self.input_data = None
        self.targets = None
        self.saved_trees = {}
        self.number_of_trees = number_of_trees

        if self.variant == BOOSTED:
            self.svm = LinearSVC()
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
        pool = Pool(processes=self.cores)

        # Python parralization preparation
        forest = self

        # Split data by pattern of nans, to optimize speed and memory
        patterns = np.unique(np.isnan(input_array), axis=0)

        optimized_input_array = [np.where((np.isnan(input_array) == pattern).all(axis=1))[0] for pattern in patterns]
        # optimized_input_array = np.array(optimized_input_array)
        parralization_input = [(forest, input_array[input_indicies]) for input_indicies in optimized_input_array]

        split_results = pool.starmap(Forest._predict_ADHOC_parralized, parralization_input)

        # Flatten the arrays
        split_results = np.hstack(np.array(split_results))
        optimized_input_array_flat = np.hstack(optimized_input_array)

        # Reorder the array so that the results match the inputs
        reorder_indicies = optimized_input_array_flat.argsort()
        results = split_results[reorder_indicies]

        return results

    @staticmethod
    def _predict_ADHOC_parralized(root_forest,
                                  input_array):  # Todo: better name than root_forest. Needs 2 differ from forest

        assert len(np.unique(np.isnan(input_array), axis=0)) == 1
        pattern = np.unique(np.isnan(input_array), axis=0)[0]

        # Select the part of the saved data that matches the pattern of missing values of the input
        matching_input = root_forest.input_data[np.where((np.isnan(root_forest.input_data) == pattern).all(axis=1))]
        matching_targets = root_forest.targets[np.where((np.isnan(root_forest.input_data) == pattern).all(axis=1))]

        if root_forest.number_of_trees == "AUTO":
            forest = RandomForestClassifier()
        else:
            forest = RandomForestClassifier(n_estimators=root_forest.number_of_trees)

        # Todo: Parameters??
        no_nan_matching_input = matching_input[:, ~pattern]
        if no_nan_matching_input.shape[1] > 0 and no_nan_matching_input.shape[0]:
            forest.fit(no_nan_matching_input, matching_targets)  #

            no_nan_input = input_array[:, ~pattern]
            result = forest.predict(no_nan_input)  # Due to single sample


        else:  # Choose from the known classes

            if len(matching_targets) > 0:
                result = np.random.choice(matching_targets, len(input_array))
            else:  # Unseen class
                result = np.random.choice(root_forest.targets, len(input_array))
        return np.array(result)

    def _fit_BOOSTED(self, input_array, targets):
        # Resample data
        log.info("Fitting has begun.")
        log.debug("Current system state: %s " % (str(psutil.virtual_memory())))
        number_of_unique_patterns = len(np.unique(np.isnan(input_array), axis=0))

        number_of_trees = self.number_of_trees
        while len(self.trees.keys()) == 0:  # catch edgecase : no viable trees could be generated todo: proper handling?

            for i in range(number_of_trees):  # Todo: Parallize
                log.info("At %d of %d trees" % (i + 1, number_of_trees))
                # Choose feature subset
                number_of_features = int(np.math.sqrt(input_array.shape[1]))  # Root of features is standard ->wiki link
                # Todo: In super-sparse data, (lot of nans) there are cases in which finding any datapoints which  even only a root amount of functioning datapoints is unlikely
                selected_features = np.random.choice(input_array.shape[1], number_of_features, replace=False)

                # Draw data where None of the features are nan, reduce features to selected
                sub_feature_array = input_array[:, selected_features]
                mask = np.isnan(sub_feature_array).any(axis=1)
                mask = np.invert(mask, dtype=bool)

                viable_input = input_array[mask][:, selected_features]
                viable_targets = targets[mask]
                if len(
                        viable_targets) == 0:  # If selected features are unusable, start over. Current solution causes less than selected number of trees to be created
                    continue

                subsampled_input = viable_input[
                    np.random.choice(range(len(viable_input)), len(viable_input))]  # Todo: a must be non empty!
                subsampled_targets = viable_targets[np.random.choice(range(len(viable_targets)), len(viable_targets))]

                # Generate trees:
                next_tree = DecisionTreeClassifier()
                next_tree.fit(subsampled_input, subsampled_targets)
                self.trees[tuple(selected_features)] = next_tree
                if len(self.trees.keys()) == 0:
                    log.warning("No viable tree was generated on this try, tying again")

        log.info("Of %d trees, %d were generated" % (number_of_trees, len(self.trees.keys())))
        log.debug("Current system state: %s " % (str(psutil.virtual_memory())))
        # Generate learner data: (samples,output_of_trees) as x, target class as y
        learner_input = self._evaluate_trees(input_array)
        assert len(learner_input) == len(targets)

        # Flatten for encoding, unflatten for learning
        input_shape = learner_input.shape
        flat_input = learner_input.flatten()
        encoder_input = flat_input
        if not "NaN" in flat_input:
            encoder_input = np.append(encoder_input, "NaN")
        self.encoder.fit(encoder_input)
        encoded_input = self.encoder.transform(flat_input)
        encoded_input = encoded_input.reshape(input_shape)

        # Train learner
        log.info("SVM fitting has begun")
        self.svm.fit(encoded_input, targets)
        log.info("Fitting finished")

    def _predict_BOOSTED(self, input_array):

        log.info("Evaluation has begun for %d inputs" % len(input_array))
        log.debug("Current system state: %s " % (str(psutil.virtual_memory())))
        learner_input = self._evaluate_trees(input_array)
        # Flatten for encoding, unflatten for learning
        input_shape = learner_input.shape
        flat_input = learner_input.flatten()
        encoded_input = self.encoder.transform(flat_input)
        encoded_input = encoded_input.reshape(input_shape)

        log.info("SVM prediction has begun")
        predictions = self.svm.predict(encoded_input)
        log.info("Evaluation has finished.")
        log.debug("Current system state: %s " % (str(psutil.virtual_memory())))

        import gc
        gc.collect()

        return predictions

    def _evaluate_trees(self, input_array) -> np.ndarray:

        packed_data = [(selected_features, tree, input_array) for selected_features, tree in self.trees.items()]

        pool = Pool(processes=self.cores)

        tree_result = pool.starmap(Forest._evaluate_trees_parralized, packed_data)

        return np.array(tree_result).transpose()

    @staticmethod
    def _evaluate_trees_parralized(selected_features, tree, input_array):
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

        return final_result

    def __str__(self):
        return "%s-Forest id: %s" % (self.variant, id(self))
