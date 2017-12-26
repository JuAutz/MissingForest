"""This module makes a number of helper functions for evaluation available."""

import numpy as np
import time
from sklearn.model_selection import KFold
from copy import deepcopy


def evaluatue(model, input_data: np.array, targets: np.array, numer_of_folds: int = 5):
    kf = KFold(n_splits=numer_of_folds)
    orignal = deepcopy(model)
    for train, test in kf.split(input_data):
        model = deepcopy(orignal)
        start = time.time()
        model.fit(input_data[train], targets[train])
        predicted_targets = model.predict(input_data[test])
        diff = time.time() - start
        hits = predicted_targets == targets[test]
        accuracy = sum(hits) / float(len(hits))
        print("Hit percentage %d in time %d for %s" % ((accuracy * 100), diff, str(model)))  # Todo: Better evaluation
