import sys

import numpy as np
sys.path.append('../')
from data_processing.read_data import read_preprocessed_data
from data_processing.split_data import SplitData
from data_processing import perform_pca
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.preprocessing import Normalizer, normalize, StandardScaler
import pandas as pd

# ANIMALS
INPUT_FILE = "../../data/animals/label_scores.txt"
# PLANTS
# INPUT_FILE = "../../data/plants/label_scores.txt"
FEATURES_FILE = '../../data/aaindex/aaindex_used.txt'


def rand_forest(labels, features):
    rf = RandomForestClassifier()
    rf.fit(features, labels)
    # rf.apply()

    return rf


def rand_forest_pca(labels, features, n_folds):
    sd = SplitData(labels, features, n_splits=n_folds)
    Ytr, Xtr = sd.Ytr, sd.Xtr
    Yte, Xte = sd.Yte, sd.Xte
    print len(Ytr), len(Ytr[0]), len(Xtr), len(Xtr[0]), len(Yte), len(Yte[0]),len(Xte), len(Xte[0])
    scores = []
    rf = RandomForestClassifier()
    for ytr, xtr, yte, xte in zip(Ytr, Xtr, Yte, Xte):
        pca = perform_pca.pca_preprocess(ytr, xtr, output_path=INPUT_FILE)
        xtr_fitted = perform_pca.tranform_data(xtr, pca)
        xte_fitted = perform_pca.tranform_data(xte, pca)
        print len(ytr), len(ytr[0]), len(xtr_fitted), len(xtr_fitted[0]), len(yte), len(yte[0]), len(xte_fitted), len(xte_fitted[0])
        rf.fit(xtr_fitted, ytr)
        scores.append(rf.score(xte_fitted, yte))
    return scores
    pass

def get_feature_vector_length(file):
    with open(file) as f:
        l = f.readline().split('|')
        vector_len = len(l)
    return vector_len


if __name__ == '__main__':
    # number of divisions for cross validation
    vec_length = get_feature_vector_length(INPUT_FILE)
    CROSS_VAL = 5
    # Y - labels, X - features
    Y, X = read_preprocessed_data(INPUT_FILE, FEATURES_FILE, CROSS_VAL)
    # x_train = pd.read_csv(INPUT_FILE, sep='|', usecols=range(1, vec_length), header=None)
    # y_train = pd.read_csv(INPUT_FILE, sep='|', usecols=[0], header=None)

    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=0)
    rf = RandomForestClassifier(n_estimators=50, criterion='gini', max_features='auto',
                                min_samples_split=2, verbose=1)
    ef = ExtraTreesClassifier(n_estimators=50, criterion='gini', max_features='auto',
                              min_samples_split=2, verbose=1)

    x_len_before = len(X[0])
    # X = normalize(X, norm='l2', axis=1)
    # X = VarianceThreshold(threshold=0.00000005).fit_transform(X)
    # X = SelectKBest(f_classif, k=50).fit_transform(X, Y)
    # X = StandardScaler().fit_transform(X)
    print('# features used: %d / %d' % (len(X[0]), x_len_before))

    print('RandomForestClassifier')
    scores = cross_val_score(rf, X, Y, cv=CROSS_VAL)
    print(scores)
    # The mean score and the 95% confidence interval
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    print('ExtraTreesClassifier')
    scores = cross_val_score(ef, X, Y, cv=CROSS_VAL)
    print(scores)
    # The mean score and the 95% confidence interval
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # print labels
    # print features
