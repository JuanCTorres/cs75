# inspired by: http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
# http://blog.datadive.net/selecting-good-features-part-iii-random-forests/
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

import pandas
import sys

sys.path.append('../')
from data_processing.read_data import read_preprocessed_data
from sklearn.preprocessing import StandardScaler


CROSS_VAL = 5
# ANIMALS
INPUT_FILE = "../../data/animals/label_scores.txt"
# PLANTS
# INPUT_FILE = "../../data/plants/label_scores.txt"
FEATURES_FILE = '../../data/aaindex/aaindex_used.txt'

dataframe = pandas.read_csv(INPUT_FILE, delimiter='|', header=None)
dataset = dataframe.values

X = dataset[:, 1:].astype(float)
y = dataset[:, 0]
# X = StandardScaler().fit_transform(X)     # makes no difference
names = list()
with open(FEATURES_FILE, 'r') as ifile:
    for line in ifile:
        names.append(line)


# Build a forest and compute the feature importances
# forest = ExtraTreesClassifier(n_estimators=250,
#                               random_state=0)
forest = RandomForestClassifier(n_estimators=25,
                                random_state=0, verbose=True)

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Print the feature ranking by name
print("Features sorted by their score:")
print(sorted(zip(map(lambda x: round(x, 6), forest.feature_importances_), names), reverse=True))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()