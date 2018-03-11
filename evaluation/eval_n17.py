import numpy as np
from datageneration import prep_data_from_file
from evaluation.helper import evaluatue
from sklearn.preprocessing import Imputer
from sklearn.tree import DecisionTreeClassifier
import Forest

if __name__ == "__main__":
    train_input, train_targets = prep_data_from_file.generate_data(prep_data_from_file.MEDICAL_N17_BINARY)

    model = Forest.Forest(variant=Forest.ADHOC)
    evaluatue(model, input_data=train_input, targets=train_targets)

    model = Forest.Forest(variant=Forest.BOOSTED,number_of_trees=10)
    evaluatue(model, input_data=train_input, targets=train_targets)

    imp=Imputer()
    train_input=imp.fit_transform(train_input)
    model=DecisionTreeClassifier()
    evaluatue(model, input_data=train_input, targets=train_targets)


