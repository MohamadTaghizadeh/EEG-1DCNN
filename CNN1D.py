import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense, BatchNormalization, Activation

#DEEP 1D-CNN
model = Sequential()
model.add(Conv1D(64, 10, activation='relu', input_shape=(640, 2), padding='same'))
model.add(BatchNormalization())
model.add(Conv1D(64, 10, activation='relu', padding='valid'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Conv1D(32, 4, activation='relu', padding='valid'))
model.add(MaxPooling1D(pool_size=3))
model.add(Flatten())
model.add(Dense(306, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(153, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(77, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(77, activation='relu'))
model.add(Dense(5, activation='softmax'))

# SETTING
cost_function = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss=cost_function, metrics=['accuracy'])

# TRAIN
epochs = 100
batch_size = 10
model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(val_data, val_labels))

# SAVE THE MODEL
model.save('pretrain_model.h5')