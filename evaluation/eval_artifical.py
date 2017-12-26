import numpy as np
from datageneration import artifical_function_data
from evaluation.helper import evaluatue
import Forest

if __name__ == "__main__":
    train_input, train_targets = artifical_function_data.generate_data(1000)
    model = Forest.Forest(variant=Forest.ADHOC)
    evaluatue(model, input_data=train_input, targets=train_targets)

    train_input, train_targets = artifical_function_data.generate_data(1000)
    model = Forest.Forest(variant=Forest.BOOSTED)
    evaluatue(model, input_data=train_input, targets=train_targets)
