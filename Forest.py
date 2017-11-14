import numpy as np
from Tree import Tree
from scipy.special import comb

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

#Subset modes
ALL="ALL"
RANDOM="RANDOM"

#Variants
ADHOC="ADHOC"
BOOSTED="BOOSTED"

from logging import Logger
log=Logger(__file__)

class Forest:

    def __init__(self, max_depth=None, cores=None, bootstrap_mode:str= ALL,resample_mode=None,variant:str=ADHOC):
        self.max_depth=max_depth
        self.cores=cores
        self.bootstrap_mode=bootstrap_mode
        self.resample_mode=resample_mode
        self.trees=[]
        self.variant=variant
        self.input_data=None
        self.targets=None
        self.saved_trees={}

    def fit(self,input_array:np.ndarray,targets:np.array):
        # Maybe not what is really needed:
        # Live computation of the models, only store the inputs.=v
        # Use the data within the input data that fits the missing value pattern,
        # bag it and predict using the resulting forest?
        if self.variant==ADHOC:
            self._fit_ADHOC(input_array,targets)
        elif self.variant==BOOSTED:
            self._fit_BOOSTED(input_array,targets)
        else:
            pass #Todo: Add the other variants

        # subsets=self._generate_subsets(input_array,targets)
        # for subset in subsets:
        #     next_tree=Tree(max_depth=self.max_depth,cores=self.cores)
        #     next_tree.fit(subset[0],subset[1])
        #     self.trees.append(next_tree)

    def predict(self,input_array:np.ndarray):
        if self.variant==ADHOC:
            return self._predict_ADHOC(input_array)
        elif self.variant== BOOSTED:
            return self._predict_BOOSTED(input_array)



    def _fit_ADHOC(self, input_array, targets):
        if self.input_data  is None and self.targets is None:
            self.input_data = input_array
            self.targets = targets
        else:
            # Todo: Should additional data be added online? No issue to implement, makes comparison iffy
            raise RuntimeError("Forest already fed with data")

    def _predict_ADHOC(self, input_array):
        #Select matching subset of data
        results=[]

        for input in input_array: #Todo: Iterate over patterns instead of inputs

            pattern=np.isnan(input)

            #Select the part of the saved data that matches the pattern of missing values of the input
            matching_input=self.input_data[np.where((np.isnan(self.input_data)==pattern).all(axis=1))]
            matching_targets=self.targets[np.where((np.isnan(self.input_data)==pattern).all(axis=1))]
            forest=RandomForestClassifier() #Todo: Parameters??
            forest.fit(matching_input,matching_targets)
            result=forest.predict([input])

            self.saved_trees[str(pattern)]=forest #Todo: Make use of saved trees
            results.append(result)

        return results

    def _fit_BOOSTED(self, input_array, targets):
        # Generate random forest (Split data into train forest/train learner?
            # Resample data
        number_of_unique_patterns=len(np.unique(np.isnan(input_array), axis=1))
        #idea: Number of trees equals to number of patterns=>
        #More complicates data requires more complicated model <==> higher number of trees.
        for i in range(number_of_unique_patterns): # Todo: Parrallize
            # Choose feature subset
            number_of_features= np.math.sqrt(input_array.shape[1]) #Root of features is standart ->wiki link
            selected_features=np.random.choice(input_array.shape[1],number_of_features,replacement=False)
            # Draw data where None of the features are nan, reduce features to selected
            sub_feature_array = input_array[:, selected_features]
            mask = np.isnan(sub_feature_array).any(axis=1)
            mask = np.invert(mask, dtype=bool)

            viable_input=input_array[mask][:,selected_features]
            viable_targets=targets[mask]

            subsampled_input = viable_input[np.random.choice(viable_input,len(viable_input))]
            subsampled_targets = viable_targets[np.random.choice(viable_targets,len(viable_targets))]

            # Generate trees:
            # Can use sklearn decison trees, must handle value error prior, and turn => turn nan error (Value Error) to special value!
            self.trees = [] # use list instead of dicts
            next_tree = DecisionTreeClassifier()
            next_tree.fit(subsampled_input,subsampled_targets)
            self.trees.append(next_tree)

        # Create and train learner

            # Connect trainer to trees

            # Train learner



    def _predict_BOOSTED(self, input_array):
        # Can use sklearn decison trees, must handle value error prior, and turn => turn nan error(Value Error ) to special value!
        pass












