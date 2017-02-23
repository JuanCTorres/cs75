import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

DATA_FILE = '../../data/plants/label_scores.txt'
FEATURE_NAMES_FILE = '../../data/aaindex/aaindex_used.txt'


def get_feature_vector_length(file):
    with open(file) as f:
        l = f.readline().split('|')
        vector_len = len(l)
    return vector_len


def get_feature_names(file):
    with open(file) as f:
        return [l.strip() for l in f.readlines()]


def dimensionality_reduction():
    vec_length = get_feature_vector_length(DATA_FILE)
    feature_names = get_feature_names(FEATURE_NAMES_FILE)
    # print(vec_length)
    # print(len(feature_names))
    x_train = pd.read_csv(DATA_FILE, sep='|', usecols=range(1, vec_length), header=None)  # , names=feature_names)
    y_train = pd.read_csv(DATA_FILE, sep='|', usecols=[0], header=None)  # , names=feature_names)
    x_test = pd.read_csv(DATA_FILE, sep='|', usecols=(1, vec_length), header=None)

    # shuffled_data = x_train.reindex(np.random.permutation(x_train.index))
    # random_forest = RandomForestClassifier()
    # random_forest.fit()
    # pca = PCA(n_components=20)
    # x_train_reduced = pca.fit(x_train).transform(x_train)
    # print(x_train_reduced)


dimensionality_reduction()
