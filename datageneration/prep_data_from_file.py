"""This module handels the reading and preparing of data from files"""
import pandas as pd
import numpy as np
from os.path import join

HEARTHSTONE = "Hearthstone"
HEARTHSTONE_FILE = join("datasets", "t-a-h-c-hearthstone.csv")


def generate_data(datasetname: str):
    if datasetname == HEARTHSTONE:
        return _generate_from_hearthstone_set()
    else:
        raise RuntimeError("Setname unknown.")


def _generate_from_hearthstone_set():
    full_frame = pd.read_csv(HEARTHSTONE_FILE)
    target_frame = full_frame["type"]
    input_frame = full_frame[["attack", "health", "cost"]]

    dataset_inputs = input_frame.values
    dataset_targets = target_frame.values
    dataset_targets = dataset_targets.astype(str)

    return dataset_inputs, dataset_targets
