import numpy as np
import pandas as pd

import time

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.preprocessing import Normalizer, normalize, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier


from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis

import sys
sys.path.append('../')
from data_processing.read_data import read_preprocessed_data


# ANIMALS
INPUT_FILE = "../../data/animals/label_scores.txt"
# PLANTS
# INPUT_FILE = "../../data/plants/label_scores.txt"
FEATURES_FILE = '../../data/aaindex/aaindex_used.txt'

VERBOSE = True
CROSS_VAL = 5

NAMES = [
    "Nearest Neighbors",
    # "Linear SVM 1"
    # "Linear SVM 2",
    # "RBF SVM",
    "SGD",
    # "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    # "AdaBoost",
    # "Naive Bayes",
    # "QDA",
    "LQA"
]

CLASSIFIERS = [
    KNeighborsClassifier(17), # ~49% acc
    # LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, # fails to converge. takes long to run
    #           multi_class='ovr', fit_intercept=True, intercept_scaling=1,
    #           class_weight=None, verbose=VERBOSE, random_state=None, max_iter=1000),
    # SVC(kernel="linear", C=0.025, verbose=VERBOSE), # ~47% acc, takes about 5 min to run on 20000 inputs
    # SVC(gamma=2, C=1, verbose=VERBOSE), # takes too long on 20000 inputs
    SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1, eta0=0.0,    # ~0.41 acc, 3 sec
                  fit_intercept=True, l1_ratio=0.15, learning_rate='optimal', loss='hinge',
                  n_iter=5, n_jobs=1, penalty='l2', power_t=0.5, random_state=None,
                  shuffle=True, verbose=VERBOSE, warm_start=False),
    # GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),     # takes too long
    DecisionTreeClassifier(max_depth=None),     # 0.41 acc, 20 sec
    RandomForestClassifier(n_estimators=10, criterion='gini', max_features='auto',      # ~0.49 acc, 12 sec
                           min_samples_split=2, verbose=VERBOSE),
    MLPClassifier(alpha=1, verbose=VERBOSE),        # 0.48 acc, 29 sec
    # AdaBoostClassifier(),       # 0.30 acc, 72 sec
    # GaussianNB(),       # 0.09 acc, 1 sec
    # QuadraticDiscriminantAnalysis(),     # 0.18 acc, 2 sec
    LinearDiscriminantAnalysis()        # 0.46 acc, 2 sec
]


def generate_tuple_lists(cla, tags):
    assert len(cla) == len(tags)
    l = list()

    for i in range(0, len(cla)):
        l.append((tags[i], cla[i]))

    return l


if __name__ == '__main__':
    Y, X = read_preprocessed_data(INPUT_FILE, FEATURES_FILE, CROSS_VAL)

    x_len_before = len(X[0])
    # X = normalize(X, norm='l2', axis=1)
    # X = VarianceThreshold(threshold=0.00000005).fit_transform(X)
    # X = SelectKBest(f_classif, k=50).fit_transform(X, Y)
    X = StandardScaler().fit_transform(X)
    print('# features used: %d / %d' % (len(X[0]), x_len_before))

    # clf1 = LogisticRegression(random_state=1, verbose=VERBOSE)
    # clf2 = RandomForestClassifier(random_state=1, verbose=VERBOSE)
    # clf3 = GaussianNB()
    #
    # vc1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],
    #                          voting='hard')
    # scores = cross_val_score(vc1, X, Y, cv=CROSS_VAL)

    estimators = generate_tuple_lists(CLASSIFIERS, NAMES)
    vc = VotingClassifier(estimators, voting='hard')
    start_time = time.time()
    scores = cross_val_score(vc, X, Y, cv=CROSS_VAL, verbose=VERBOSE)
    end_time = time.time()

    print(scores)
    # The mean score and the 95% confidence interval

    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    print("--- %s seconds ---" % (end_time - start_time))

    # Curr best: 100k samples, cv=5
    # Accuracy: 0.52(+ / - 0.07)
    # 814.231223106 seconds




