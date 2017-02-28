# inspired by: http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
# http://blog.datadive.net/selecting-good-features-part-iii-random-forests/
import numpy as np
import matplotlib
import sys

matplotlib.use('TKAgg')

import matplotlib.pyplot as plt
import pandas as pd

from data_processing.read_data import read_preprocessed_data
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

INPUT_FILE = "../../data/plants/label_scores.txt"
FEATURES_FILE = '../../data/aaindex/aaindex_used.txt'
CROSS_VAL = 5
DEBUG = True


def plot_feature_importance():
    max_args = 40  # max number of features to include in plot.
    sys.path.append('../')
    # ANIMALS
    # INPUT_FILE = "../../data/animals/label_scores.txt"
    # PLANTS

    dataframe = pd.read_csv(INPUT_FILE, delimiter='|', header=None)
    dataset = dataframe.values

    X = dataset[:, 1:].astype(float)
    y = dataset[:, 0]
    # X = StandardScaler().fit_transform(X)     # makes no difference
    with open(FEATURES_FILE, 'r') as ifile:
        names = [line for line in ifile]

    # Build a forest and compute the feature importances
    # forest = ExtraTreesClassifier(n_estimators=250,
    #                               random_state=0)
    forest = RandomForestClassifier(n_estimators=25,
                                    random_state=0, verbose=True)

    forest.fit(X, y)
    importance_list = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importance_list)[::-1]
    with open(FEATURES_FILE, 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    sorted_feature_names = [feature_names[i] for i in indices]

    if DEBUG:
        # Print the feature ranking
        print('Feature ranking:')
        for f in range(X.shape[1]):
            print("%d. feature %d (%f)" % (f + 1, indices[f], importance_list[indices[f]]))

        # Print the feature ranking by name
        print("Features sorted by their score:")
        print(sorted(zip(map(lambda x: round(x, 6), forest.feature_importances_), names), reverse=True))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title('Feature importance')
    grid = range(X.shape[1] + 1)
    plt.bar(grid[: max_args + 1], importance_list[indices[: max_args + 1]],
            color='r', yerr=std[indices[: max_args + 1]], align='center')

    plt.xticks(range(max_args + 1), sorted_feature_names[: max_args + 1], rotation='vertical')
    # plt.xlim([-1, X.shape[1]])
    plt.show()


if __name__ == '__main__':
    plot_feature_importance()
