import os
import sys
sys.path.append("/root")
import CNN1D
import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
from processor import Utils
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle
from sklearn.preprocessing import minmax_scale

tf.autograph.set_verbosity(0)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

source_path = "../eegpaper/"
save_path = os.path.join("../saved_models", "roi")
os.mkdir(save_path)

# Load data
channels = Utils.combinations["a"] #["FC1", "FC2"], ["FC3", "FC4"], ["FC5", "FC6"]]
#channels = Utils.combinations["b"]
#channels = Utils.combinations["c"]
#channels = Utils.combinations["d"]
#channels = Utils.combinations["e"]
#channels = Utils.combinations["f"]

exclude =  [38, 88, 89, 92, 100, 104]
subjects = [n for n in np.arange(1,110) if n not in exclude]
x, y = Utils.load(channels, subjects, base_path=source_path)
y_one_hot  = Utils.to_one_hot(y, by_sub=False)
reshaped_x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
x_train_raw, x_valid_test_raw, y_train_raw, y_valid_test_raw = train_test_split(reshaped_x,
                                                                            y_one_hot,
                                                                            stratify=y_one_hot,
                                                                            test_size=0.20,
                                                                            random_state=42)

x_train_scaled_raw = minmax_scale(x_train_raw, axis=1)
x_test_valid_scaled_raw = minmax_scale(x_valid_test_raw, axis=1)

x_valid_raw, x_test_raw, y_valid, y_test = train_test_split(x_test_valid_scaled_raw,
                                                    y_valid_test_raw,
                                                    stratify=y_valid_test_raw,
                                                    test_size=0.50,
                                                    random_state=42)

x_valid = x_valid_raw.reshape(x_valid_raw.shape[0], int(x_valid_raw.shape[1]/2),2).astype(np.float64)
x_test = x_test_raw.reshape(x_test_raw.shape[0], int(x_test_raw.shape[1]/2),2).astype(np.float64)

print ('before oversampling = {}'.format(y_train_raw.sum(axis=0)))
sm = SMOTE(random_state=42)
x_train_smote_raw, y_train = sm.fit_resample(x_train_scaled_raw, y_train_raw)
print ('before oversampling = {}'.format(y_train_raw.sum(axis=0)))
print ('after oversampling = {}'.format(y_train.sum(axis=0)))

x_train = x_train_smote_raw.reshape(x_train_smote_raw.shape[0], int(x_train_smote_raw.shape[1]/2), 2).astype(np.float64)


learning_rate = 1e-4

loss = tf.keras.losses.categorical_crossentropy
optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
model = CNN1D()
modelPath = os.path.join(os.getcwd(), 'bestModel.h5')

model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

checkpoint = ModelCheckpoint( 
    modelPath, 
    monitor='val_loss', 
    verbose=1, 
    save_best_only=False, 
    save_weights_only=False, 
    mode='auto', 
    period=1 
    )


hist = model.fit(x_train, y_train, epochs=100, batch_size=10,
                validation_data=(x_valid, y_valid), callbacks=callbacksList)

with open(os.path.join(save_path, "hist.pkl"), "wb") as file:
    pickle.dump(hist.history, file)

model.save(save_path)

"""
Test
"""

del model
model = tf.keras.models.load_model(save_path, custom_objects={"CustomModel": CNN1D})

testLoss, testAcc = model.evaluate(x_test, y_test)
print('\nAccuracy:', testAcc)
print('\nLoss: ', testLoss)

yPred = model.predict(x_test)

yTestClass = np.argmax(y_test, axis=1)
yPredClass = np.argmax(yPred,axis=1)

print('\n Classification report \n\n',
  classification_report(
      yTestClass,
      yPredClass,
       target_names=["B", "R", "RL", "L", "F"]
      )
  )

print('\n Confusion matrix \n\n',
  confusion_matrix(
      yTestClass,
      yPredClass,
      )
  )
