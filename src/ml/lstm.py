import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import time

print("LOADING DATA...")
# load ascii text and covert to lowercase
amino_acids = "0ALRKNMDFCPQSETGWHYIVXBZJUO"
locations = ['Vacuole', 'Golgi apparatus', 'Secreted', 'Cytoplasm', 'Mitochondria', 'Peroxisome', 'Nucleus', 'GPI anchored', 'Membrane', 'Cytoskeleton', 'Lysosome', 'Plasma membrane', 'ER']
aa_dict = dict((aa, i) for i, aa in enumerate(amino_acids))
location_dict = dict((aa, i) for i, aa in enumerate(locations))
sequence_file = "../../data/animals/label_sequences.txt"
# input_data = open(sequence_file)
# sequence_len, sequence_num = 500,  1100000
# X_data, Y_data = np.zeros([sequence_num, 1, sequence_len]), np.zeros(sequence_num)
# for l_index, line in enumerate(input_data):
# 	data = line.strip('\n').split("|")
# 	location, sequence = data[0], data[1]
# 	sequence_vector = np.array([aa_dict[element] for element in sequence])
# 	if len(sequence_vector)>500:
# 		X_data[l_index,0,:] = sequence_vector[:500]
# 	else:
# 		X_data[l_index,0,:] = np.concatenate((sequence_vector,np.zeros(sequence_len-len(sequence_vector))))
# 	Y_data[l_index] = location_dict[location]

# print("INPUT COMPLETE")
# np.savez("data_set.npz", X_data, Y_data)
#

data = np.load("data_set.npz")
X_data = data['arr_0']
Y_data = data['arr_1']
print("LOADING DATA COMPLETE. INITIALIZING MODEL...")
# normalize
X_data = X_data / float(27)

# # one hot encode the output variable
Y_data = np_utils.to_categorical(Y_data)

# # define the LSTM model
def build_model():
	model = Sequential()
	model.add(LSTM(256, input_shape=(X_data.shape[1], X_data.shape[2]), return_sequences=True))
	model.add(Dropout(0.2))
	model.add(LSTM(256))
	model.add(Dropout(0.2))
	model.add(Dense(Y_data.shape[1], activation='softplus'))
	model.compile(loss='categorical_crossentropy', optimizer='adamax')
	# define the checkpoint
	filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
	callbacks_list = [checkpoint]
	return model


estimator = KerasClassifier(build_fn=build_model, nb_epoch=10, batch_size=100, verbose=1)

# X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.33, random_state=seed)
# estimator.fit(X_train, Y_train)
# predictions = estimator.predict(X_test)
# print(predictions)
# print(encoder.inverse_transform(predictions))

kfold = KFold(n_splits=5, shuffle=True)
print("MODEL INITIALIZATION COMPLETE. FITTING MODEL...")

start_time = time.time()
results = cross_val_score(estimator, X_data, Y_data, cv=kfold)
end_time = time.time()

print("\nBaseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
print("--- %s seconds ---" % (end_time - start_time))
# fit the model
model.fit(X_data, Y_data, nb_epoch=10, batch_size=64, callbacks=callbacks_list)
print("MODEL FITTING COMPLETE")
