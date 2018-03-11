"""This module handels the reading and preparing of data from files"""
import pandas as pd
import numpy as np
from os.path import join
from sklearn.preprocessing import LabelEncoder,MinMaxScaler


HEARTHSTONE = "Hearthstone"
HEARTHSTONE_FILE = join("datasets", "t-a-h-c-hearthstone.csv")
MEDICAL_N17_NONBINARY= "Medical_N17_Nonbinary"
MEDICAL_N17_BINARY = "Medical_N17_Binary"
BAlANCED_N17_NONBINARY= "Balanced_N17_Nonbinary"
BALANCED_N17_BINARY = "Balanced_N17_Binary"

COMPLETE_N17_FILE = join("datasets", "complete_data_n17.csv")
BAlANCED_N17_FILE = join("datasets", "balanced_data_n17.csv")






def generate_data(datasetname: str):
    if datasetname == HEARTHSTONE:
        return _generate_from_hearthstone_set()
    elif datasetname== MEDICAL_N17_NONBINARY:
        return  _generate_from_medical_set("N17_Nonbinary")
    elif datasetname== MEDICAL_N17_BINARY:
        return  _generate_from_medical_set("N17_Binary")
    elif datasetname== BAlANCED_N17_NONBINARY:
        return  _generate_from_medical_set("Balanced_N17_Nonbinary")
    elif datasetname== BALANCED_N17_BINARY:
        return  _generate_from_medical_set("Balanced_N17_Binary")
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
    if set_name=="N17_Nonbinary" or set_name == "N17_Binary":
        file=COMPLETE_N17_FILE
    elif set_name=="Balanced_N17_Nonbinary" or set_name == "Balanced_N17_Binary":
        file=BAlANCED_N17_FILE

    full_frame=pd.read_csv(file)
    if  "Binary" in set_name:
        target_frame=full_frame["icd_binary"]
    elif "Nonbinary" in set_name:
        target_frame=full_frame["icd_three"]

    input_frame=full_frame.copy()
    del input_frame["icd_three"]
    del input_frame["record_id"]
    del input_frame["icd_binary"]
    input_frame["geschlecht"]=LabelEncoder().fit_transform(input_frame["geschlecht"])

    dataset_inputs = input_frame.values

    #Normalization
    dataset_inputs=(dataset_inputs-np.nanmin(dataset_inputs))/(np.nanmax(dataset_inputs) - np.nanmin(dataset_inputs))

    dataset_targets = target_frame.values
    dataset_targets = dataset_targets.astype(str)

    return dataset_inputs, dataset_targets
