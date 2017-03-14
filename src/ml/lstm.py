import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import np_utils
import time
from keras.models import load_model

print("LOADING DATA...")
# load ascii text and covert to lowercase
amino_acids = "0ALRKNMDFCPQSETGWHYIVXBZJUO"
locations = ['Vacuole', 'Golgi apparatus', 'Secreted', 'Cytoplasm', 'Mitochondria', 'Peroxisome', 'Nucleus', 'GPI anchored', 'Membrane', 'Cytoskeleton', 'Lysosome', 'Plasma membrane', 'ER']
aa_dict = dict((aa, i) for i, aa in enumerate(amino_acids))
location_dict = dict((aa, i) for i, aa in enumerate(locations))
sequence_file = "../../data/animals/label_sequences.txt"

#############################################################################################################################
# uncommented to generate the data file
# input_data = open(sequence_file)
# sequence_len, sequence_num= 500, 1100000
# X_data, Y_data = np.zeros([sequence_num, 1, sequence_len]), np.zeros(sequence_num)
# for l_index, line in enumerate(input_data):
#     data = line.strip('\n').split("|")
#     location, sequence = data[0], data[1]
#     sequence_vector = np.array([aa_dict[element] for element in sequence])
#     if len(sequence_vector) > 500:
#         X_data[l_index, 0, :] = sequence_vector[:500]
#     else:
#         X_data[l_index, 0, :] = np.concatenate((sequence_vector, np.zeros(sequence_len-len(sequence_vector))))
#     Y_data[l_index] = location_dict[location]
#
# print("INPUT COMPLETE")
# np.savez("../../data/animals/LSTM_DATA.npz", X_data, Y_data)
#############################################################################################################################

data = np.load("../../data/animals/LSTM_DATA.npz")
X_data = data['arr_0']
Y_data = data['arr_1']
print("LOADING DATA COMPLETE. INITIALIZING MODEL...")
# normalize
X_data = X_data / float(27)

# # one hot encode the output variable
Y_data = np_utils.to_categorical(Y_data)

# model = load_model('../../models/weights-improvement-15-1.1306.hdf5')

# # define the LSTM model
model = Sequential()
model.add(LSTM(500, input_shape=(X_data.shape[1], X_data.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(500, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(500))
model.add(Dropout(0.2))
model.add(Dense(Y_data.shape[1], activation='softplus'))
model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['accuracy'])

# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
earlystop = EarlyStopping(monitor='loss', min_delta=1e-5, patience=0, verbose=1, mode='auto')
callbacks_list = [checkpoint, earlystop]

print("MODEL INITIALIZATION COMPLETE. FITTING MODEL...")
start_time = time.time()
model.fit(X_data[:1000000], Y_data[:1000000], nb_epoch=50, batch_size=1000, verbose=1, callbacks=callbacks_list)
end_time = time.time()

print("MODEL FITTING COMPLETE. SAVING MODEL...")
model.save('../../models/LSTM2L50E1kB.hdf5')
print("SAVING MODEL COMPLETE. EVALUATING MODEL...")

scores = model.evaluate(X_data[1000000:], Y_data[1000000:], verbose=1)
print("\nBaseline: %.2f%%" % float(scores[1]*100))
print("--- %s seconds ---" % (end_time - start_time))
