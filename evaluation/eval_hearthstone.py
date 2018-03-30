import numpy as np
from datageneration import prep_data_from_file
from evaluation.helper import evaluatue_detailed_confusion_matrix
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
import pickle
import Forest
ORDER=["SPELL","MINION","ENCHANTMENT","HERO","HERO_POWER","WEAPON"]

if __name__ == "__main__":
    train_input, train_targets = prep_data_from_file.generate_data(prep_data_from_file.HEARTHSTONE)


    model = Forest.Forest(variant=Forest.ADHOC)
    result_ADHOC = evaluatue_detailed_confusion_matrix(model,ORDER, input_data=train_input, targets=train_targets)

    model = Forest.Forest(variant=Forest.BOOSTED)
    result_BOOSTED = evaluatue_detailed_confusion_matrix(model,ORDER, input_data=train_input, targets=train_targets)

    imp = Imputer()
    train_input = imp.fit_transform(train_input)
    model = RandomForestClassifier()
    result_IMPUTED = evaluatue_detailed_confusion_matrix(model,ORDER, input_data=train_input, targets=train_targets)

    total_result=[result_ADHOC,result_BOOSTED,result_IMPUTED]
    with open("results_hearthstone.p","xb") as file:
        pickle.dump(total_result,file)

