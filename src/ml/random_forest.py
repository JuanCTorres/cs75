import sys
sys.path.append('../')
from data_processing.read_data import read_preprocessed_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.preprocessing import Normalizer, normalize


def rand_forest(labels, features):
    rf = RandomForestClassifier()
    rf.fit(features, labels)
    # rf.apply()

    return rf


if __name__ == '__main__':
    input_file = "../../data/plants/label_scores.txt"

    # number of divisions for cross validation
    cv = 5
    # Y - labels, X - features
    Y, X = read_preprocessed_data(input_file, cv)
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=0)
    rf = RandomForestClassifier(n_estimators=10, criterion='gini', max_features='auto',
                                min_samples_split=2, verbose=1)

    X = normalize(X, norm='l2', axis=1, copy=False)
    print('# features before var filter: %d'% len(X[0]))
    # X = VarianceThreshold(threshold=0.00000005).fit_transform(X)
    print('# features after var filter: %d' % len(X[0]))
    X = SelectKBest(f_classif, k=50).fit_transform(X, Y)



    scores = cross_val_score(rf, X, Y, cv=cv)

    print(scores)
    # The mean score and the 95% confidence interval
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



    # print labels
    # print features
