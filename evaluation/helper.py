"""This module makes a number of helper functions for evaluation available."""

import numpy as np
import time
from sklearn.model_selection import KFold
from copy import deepcopy
from logging import Logger
import psutil
from sklearn.metrics import confusion_matrix
LOG=Logger(__file__)

def evaluatue(model, input_data: np.array, targets: np.array, numer_of_folds: int = 5):
    kf = KFold(n_splits=numer_of_folds, shuffle=True)
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

def evaluatue_detailed(model, input_data: np.array, targets: np.array, numer_of_folds: int = 5):
    kf = KFold(n_splits=numer_of_folds, shuffle=True)
    orignal = deepcopy(model)
    runtimes=[]
    accuracies=[]
    memories=[]
    for train, test in kf.split(input_data):

        memory_start=psutil.virtual_memory()
        model = deepcopy(orignal)
        start = time.time()
        model.fit(input_data[train], targets[train])
        predicted_targets = model.predict(input_data[test])
        diff_time = time.time() - start
        diff_memory=memory_start.free-psutil.virtual_memory().free
        hits = predicted_targets == targets[test]
        accuracy = sum(hits) / float(len(hits))
        accuracies.append(accuracy)
        runtimes.append(diff_time)
        memories.append(diff_memory)
        del(model)

    return str(orignal),accuracies,runtimes,memories


def evaluatue_detailed_confusion_matrix(model,labelorder, input_data: np.array, targets: np.array, numer_of_folds: int = 5):
    kf = KFold(n_splits=numer_of_folds, shuffle=True)
    orignal = deepcopy(model)
    runtimes=[]
    accuracies=[]
    memories=[]
    matrixes=[]
    for train, test in kf.split(input_data):

        memory_start=psutil.virtual_memory()
        model = deepcopy(orignal)
        start = time.time()
        model.fit(input_data[train], targets[train])
        predicted_targets = model.predict(input_data[test])
        diff_time = time.time() - start
        diff_memory=memory_start.free-psutil.virtual_memory().free
        hits = predicted_targets == targets[test]
        accuracy = sum(hits) / float(len(hits))
        matrix=confusion_matrix(targets[test],predicted_targets)
        accuracies.append(accuracy)
        runtimes.append(diff_time)
        memories.append(diff_memory)
        matrixes.append(matrix)
        del(model)

    return str(orignal), accuracies, runtimes, memories,matrixes