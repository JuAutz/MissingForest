"""This module handels the reading and preparing of data from files"""
import pandas as pd
import numpy as np
from os.path import join
from sklearn.preprocessing import LabelEncoder


HEARTHSTONE = "Hearthstone"
HEARTHSTONE_FILE = join("datasets", "t-a-h-c-hearthstone.csv")
MEDICAL_N17="Medical_N17"
N17_FILE = join("datasets", "complete_data_n17.csv")




def generate_data(datasetname: str):
    if datasetname == HEARTHSTONE:
        return _generate_from_hearthstone_set()
    elif datasetname== MEDICAL_N17:
        return  _generate_from_medical_set("N17")
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


def _generate_from_medical_set(set_name:str):
    if set_name=="N17":
        file=N17_FILE

    full_frame=pd.read_csv(file)
    target_frame=full_frame["icd_three"]
    input_frame=full_frame.copy()
    del input_frame["icd_three"]
    del input_frame["record_id"]
    input_frame["geschlecht"]=LabelEncoder().fit_transform(input_frame["geschlecht"])

    dataset_inputs = input_frame.values
    dataset_targets = target_frame.values
    dataset_targets = dataset_targets.astype(str)

    return dataset_inputs, dataset_targets
