from sklearn.datasets import load_digits as load_data
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.preprocessing import Normalizer, normalize, StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import SGDClassifier

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
# This is all that's needed for scikit-plot
import matplotlib.pyplot as plt
from scikitplot import classifier_factory
from scikitplot import plotters as skplt
import pandas
import sys

sys.path.append('../')
from data_processing.read_data import read_preprocessed_data


VERBOSE = 0
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
X = StandardScaler().fit_transform(X)

NAMES = [
    "Nearest Neighbors",
    "SGD",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "LQA"
]

CLASSIFIERS = [
    KNeighborsClassifier(17), # ~49% acc
    SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1, eta0=0.0,    # ~0.41 acc, 3 sec
                  fit_intercept=True, l1_ratio=0.15, learning_rate='optimal', loss='log',
                  n_iter=5, n_jobs=1, penalty='l2', power_t=0.5, random_state=None,
                  shuffle=True, verbose=VERBOSE, warm_start=False),
    DecisionTreeClassifier(max_depth=None),     # 0.41 acc, 20 sec
    RandomForestClassifier(n_estimators=10, criterion='gini', max_features='auto',      # ~0.49 acc, 12 sec
                           min_samples_split=2, verbose=VERBOSE),
    MLPClassifier(alpha=1, verbose=VERBOSE),        # 0.48 acc, 29 sec
    LinearDiscriminantAnalysis()        # 0.46 acc, 2 sec
]

def generate_tuple_lists(cla, tags):
    assert len(cla) == len(tags)
    l = list()

    for i in range(0, len(cla)):
        l.append((tags[i], cla[i]))

    return l

estimators = generate_tuple_lists(CLASSIFIERS, NAMES)


# ROC CURVE
nb = VotingClassifier(estimators, voting='soft')
classifier_factory(nb)
nb.plot_roc_curve(X, y, random_state=1)
plt.show()

# Feature importance
# rf = RandomForestClassifier(verbose=True)
# rf.fit(X, y)
# skplt.plot_feature_importances(rf, feature_names=X.dtypes.index, max_num_features=123)
# plt.show()
