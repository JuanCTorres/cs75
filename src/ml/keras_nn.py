import sys
sys.path.append('../')

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.preprocessing import Normalizer, normalize
from keras.optimizers import SGD, RMSprop, Adagrad, Adam, Adamax, Nadam
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
import numpy
import pandas
import math
import time


# ANIMALS
INPUT_FILE = "../../data/animals/label_scores.txt"
# PLANTS
# INPUT_FILE = "../../data/plants/label_scores.txt"


RANDSEED = 7
numpy.random.seed(RANDSEED)
# load dataset

dataframe = pandas.read_csv(INPUT_FILE, delimiter='|', header=None)
dataset = dataframe.values

X = dataset[:, 1:].astype(float)
Y = dataset[:, 0]

x_len_before = len(X[0])
# X = normalize(X, norm='l2', axis=1)
# X = VarianceThreshold(threshold=0.00000005).fit_transform(X)
# X = SelectKBest(f_classif, k=50).fit_transform(X, Y)
# X = StandardScaler().fit_transform(X)
print('# features used: %d / %d' % (len(X[0]), x_len_before))

in_dim = len(X[0])
out_dim = len(set(Y))


def basic_model():
    # create model
    model = Sequential()

    model.add(Dense(in_dim, input_dim=in_dim, init='normal', activation='softplus'))
    model.add(Dense((in_dim+out_dim)/2, init='normal', activation='softplus'))
    model.add(Dense(math.sqrt(in_dim*out_dim), init='normal', activation='softplus'))
    model.add(Dense(out_dim, init='normal', activation='softplus'))

    # optimizeers
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    rms = RMSprop(lr=0.005, rho=0.9, epsilon=1e-08, decay=0.0)
    adagrad = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    adamax = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=adamax, metrics=['accuracy'])
    return model


# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
estimator = KerasClassifier(build_fn=basic_model, nb_epoch=5, batch_size=5, verbose=1)

# X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.33, random_state=seed)
# estimator.fit(X_train, Y_train)
# predictions = estimator.predict(X_test)
# print(predictions)
# print(encoder.inverse_transform(predictions))

kfold = KFold(n_splits=5, shuffle=True)
start_time = time.time()

results = cross_val_score(estimator, X, dummy_y, cv=kfold)
end_time = time.time()

print("\nBaseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
print("--- %s seconds ---" % (end_time - start_time))

# 58.30% (0.38%) with top10