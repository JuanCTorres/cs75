from sklearn.datasets import load_digits as load_data
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# This is all that's needed for scikit-plot
import matplotlib.pyplot as plt
from scikitplot import classifier_factory
from scikitplot import plotters as skplt
import pandas
import sys

sys.path.append('../')
from data_processing.read_data import read_preprocessed_data

CROSS_VAL = 5
# ANIMALS
INPUT_FILE = "../../data/animals/label_scores.txt"
# PLANTS
# INPUT_FILE = "../../data/plants/label_scores.txt"
FEATURES_FILE = '../../data/aaindex/aaindex_used.txt'

dataframe = pandas.read_csv(INPUT_FILE, delimiter='|', header=None)
dataset = dataframe.values

# X = dataset[:, 1:].astype(float)
# y = dataset[:, 0]
# X = StandardScaler().fit_transform(X)

y, X = read_preprocessed_data(INPUT_FILE, FEATURES_FILE, CROSS_VAL, format='df')
print X.dtypes.index

# ROC CURVE
# nb = GaussianNB()
# classifier_factory(nb)
# nb.plot_roc_curve(X, y, random_state=1)
# plt.show()

# Feature importance
rf = RandomForestClassifier(verbose=True)
rf.fit(X, y)
skplt.plot_feature_importances(rf, feature_names=X.dtypes.index, max_num_features=123)
plt.show()
