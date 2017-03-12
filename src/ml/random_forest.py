import sys

sys.path.append('../')
from data_processing.read_data import read_preprocessed_data
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
