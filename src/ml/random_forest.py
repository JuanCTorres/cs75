import sys
sys.path.append('../')
from data_processing.read_data import read_preprocessed_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

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
    rf = RandomForestClassifier()

    scores = cross_val_score(rf, X, Y, cv=cv)

    print(scores)
    # The mean score and the 95% confidence interval
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



    # print labels
    # print features
