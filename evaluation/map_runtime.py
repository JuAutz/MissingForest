#!/usr/bin/python
import numpy as np
import psutil
import pympler.tracker
import sys

import sys
sys.path.append("..") #hack for using this as a script
from datageneration import artifical_function_data
from evaluation.helper import evaluatue_detailed
from sklearn.preprocessing import Imputer
from sklearn.tree import DecisionTreeClassifier
from time import time
import pickle
import itertools
import Forest
import gc
from pathlib import Path
import copy
import os

intermediate_result_file = "i_result.p"
final_result_file = "result.p"
todo_list_file = "todo_list.p"


def map_runtime():
    # Variables to iterate over Todo

    if Path(intermediate_result_file).exists():
        with open(intermediate_result_file, "rb") as file:
            result_dict = pickle.load(file)

    else:
        result_dict = {}

    if Path(todo_list_file).exists():
        with open(todo_list_file, "rb") as file:
            todo_list = pickle.load(file)

    else:
        number_of_trees = [1, 5, 10, 20, 50]
        number_of_datapoints = [50, 100, 1000, 10000, 100000]
        number_of_dimensions = [1, 2, 5, 10, 15]
        number_of_nan_percentages = [0, 0.1, 0.2, 0.5,0.7, 0.9]
        todo_list = itertools.product(number_of_trees, number_of_datapoints,
                                      number_of_dimensions,
                                      number_of_nan_percentages)
        todo_list = list(todo_list)

        with open(todo_list_file, "xb") as file:
            pickle.dump(todo_list, file)

    copy_todo_list = copy.copy(list(todo_list))
    for trees, datapoints, dimensions, nan_percentage in todo_list:

        print(psutil.virtual_memory())
        while (psutil.virtual_memory().percent > 70):
            print("Memory Leaks Critcal")
            raise  Exception("Memory Leaks Critcal")
        print(trees, datapoints, dimensions, nan_percentage)
        # Evalute Adhoc
        training_data, target_data = artifical_function_data.generate_fillerdata(datapoints, dimensions, nan_percentage)

        model = Forest.Forest(variant=Forest.ADHOC, number_of_trees=trees, cores=4)
        evaluationAD = evaluatue_detailed(model, input_data=training_data, targets=target_data)
        print(evaluationAD)

        # Evaluate Boosted
        model = Forest.Forest(variant=Forest.BOOSTED, number_of_trees=trees, cores=4)
        evaluationBO = evaluatue_detailed(model, input_data=training_data, targets=target_data)
        print(evaluationBO)

        result_dict[tuple(copy_todo_list[0])] = (evaluationAD, evaluationBO)
        if Path(intermediate_result_file).exists():
            with open(intermediate_result_file, "wb") as file:
                pickle.Pickler(file).dump(result_dict)
        else:
            with open(intermediate_result_file, "xb") as file:
                pickle.Pickler(file).dump(result_dict)

        del (copy_todo_list[0])
        with open(todo_list_file, "wb") as file:
            pickle.Pickler(file).dump(list(copy_todo_list))

        print("Files saved")

        del (model)
        del (evaluationAD)
        del (evaluationBO)

def restart():
    import sys
    print("argv was", sys.argv)
    print("sys.executable was", sys.executable)
    print("restart now")

    import os
    os.execv(sys.executable, ['python'] + sys.argv)

def auto_restart():
    try:
        map_runtime()
    except Exception:
        restart()


if __name__=="__main__":
    auto_restart()