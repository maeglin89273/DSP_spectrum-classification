import numpy as np
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical

DATA_DIR = 'sound_data'

train_x = np.load(DATA_DIR + '/sound_X_train.npy').reshape((-1, 32, 32, 1))
train_y = to_categorical(np.load(DATA_DIR + '/sound_y_train.npy'), 20)
val_x = np.load(DATA_DIR + '/sound_X_val.npy').reshape((-1, 32, 32, 1))
val_y = to_categorical(np.load(DATA_DIR + '/sound_y_val.npy'), 20)
test_x = np.load(DATA_DIR + '/sound_X_test.npy').reshape((-1, 32, 32, 1))


model = Sequential()

model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(32, 32, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(20))

adam = Adam()

model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=20, batch_size=128, validation_data=(val_x, val_y))