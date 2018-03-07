import pandas as pd
import numpy as np


training = pd.read_csv('exerciseData/trainingSetGenre.csv')
test = pd.read_csv('exerciseData/testSetGenre.csv')


print(training.head().data_base_index)
print(test.head().data_base_index)


trainingIndex = training.data_base_index.values
testIndex = test.data_base_index.values


if any([i in trainingIndex for i in testIndex]):
    print("Test and training data are mixed")
else:
    print("Training data is OK")