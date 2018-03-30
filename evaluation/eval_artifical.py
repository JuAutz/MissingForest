import numpy as np
from datageneration import artifical_function_data
from evaluation.helper import evaluatue_detailed_confusion_matrix
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
import pickle
import Forest
ORDER=["A","B","C"]

if __name__ == "__main__":
    train_input, train_targets = artifical_function_data.generate_data(1000)

    model = Forest.Forest(variant=Forest.ADHOC)
    result_ADHOC = evaluatue_detailed_confusion_matrix(model,ORDER, input_data=train_input, targets=train_targets)

    model = Forest.Forest(variant=Forest.BOOSTED)
    result_BOOSTED = evaluatue_detailed_confusion_matrix(model,ORDER, input_data=train_input, targets=train_targets)

    imp = Imputer()
    train_input = imp.fit_transform(train_input)
    model = RandomForestClassifier()
    result_IMPUTED = evaluatue_detailed_confusion_matrix(model,ORDER, input_data=train_input, targets=train_targets)

    total_result=[result_ADHOC,result_BOOSTED,result_IMPUTED]
    with open("results_artificial.p","xb") as file:
        pickle.dump(total_result,file)

