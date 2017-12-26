import numpy as np
from datageneration import prep_data_from_file
from evaluation.helper import evaluatue
import Forest

if __name__ == "__main__":
    train_input, train_targets = prep_data_from_file.generate_data(prep_data_from_file.HEARTHSTONE)
    model = Forest.Forest(variant=Forest.ADHOC)
    evaluatue(model, input_data=train_input, targets=train_targets)

    train_input, train_targets = prep_data_from_file.generate_data(prep_data_from_file.HEARTHSTONE)
    model = Forest.Forest(variant=Forest.BOOSTED)
    evaluatue(model, input_data=train_input, targets=train_targets)
