""" The functions in this module handel generation of artificial data based on a number of functions."""

import numpy as np
import pandas as pd

def generate_fillerdata(number_of_datapoints:int, number_of_dims:int,nan_percentage:float)->(np.array, np.array):
    data=pd.DataFrame(np.random.rand(number_of_datapoints,number_of_dims))

    data[data<=nan_percentage]=np.nan

    targets= np.random.rand(number_of_datapoints)
    targets=targets>0.5
    targets=targets.astype(str)
    return  data.as_matrix(),targets



def generate_data(number_of_datapoints, functions=["A", 'B', "C"]) -> (np.array, np.array):
    dataset_inputs = np.array([])
    dataset_targets = np.array([], dtype=object)
    for function in functions:
        # Generate input required for function

        function_pointer, number_of_required_inputs = _select_function(function)
        input_list = []
        for i in range(number_of_required_inputs):
            next_input = np.random.rand(int(number_of_datapoints / len(functions)))
            input_list.append(next_input)

        output_array = function_pointer(*input_list)
        # Fill with nans
        for i in range(4 - number_of_required_inputs):
            next_input = np.ones(int(number_of_datapoints / len(functions)))
            next_input.fill(np.nan)
            input_list.append(next_input)
        input_list.append(output_array)

        input_matrix = np.array(input_list).transpose()
        targets = np.ones(int(number_of_datapoints / len(functions)), dtype=object)
        targets.fill(function)

        # Make dataset out of function
        if len(dataset_inputs) < 1:
            dataset_inputs = input_matrix
            dataset_targets = targets
        else:
            dataset_inputs = np.append(dataset_inputs, input_matrix, axis=0)
            dataset_targets = np.append(dataset_targets, targets)
    return dataset_inputs, dataset_targets


def _select_function(letter):
    if letter == "A":
        return _function_A, 2
    if letter == "B":
        return _function_B, 3
    if letter == "C":
        return _function_C, 4


def _function_A(x1, x2):
    """A simple linear function."""
    return 2 * x1 + x2


def _function_B(x1, x2, x3):
    """A sinus based function"""
    return np.sin(x1) * (x2 + x3) / 2


def _function_C(x1, x2, x3, x4):
    """ A slightly more complicated polynominal function"""
    return (x1 * x2 - x3) / x4
