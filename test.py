import os
import sys
sys.path.append("..")
import CNN1D
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import minmax_scale

tf.autograph.set_verbosity(0)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

save_path = os.path.join("../saved_model", "roi")

x_train_raw, x_valid_test_raw, y_train_raw, y_valid_test_raw = train_test_split(reshaped_x,
                                                                            y_one_hot,
                                                                            stratify=y_one_hot,
                                                                            test_size=0.20,
                                                                            random_state=42)

#Scale indipendently train/test
x_train_scaled_raw = minmax_scale(x_train_raw, axis=1)
x_test_valid_scaled_raw = minmax_scale(x_valid_test_raw, axis=1)

#Create Validation/test
x_valid_raw, x_test_raw, y_valid, y_test = train_test_split(x_test_valid_scaled_raw,
                                                    y_valid_test_raw,
                                                    stratify=y_valid_test_raw,
                                                    test_size=0.50,
                                                    random_state=42)

x_valid = x_valid_raw.reshape(x_valid_raw.shape[0], int(x_valid_raw.shape[1]/2),2).astype(np.float64)
x_test = x_test_raw.reshape(x_test_raw.shape[0], int(x_test_raw.shape[1]/2),2).astype(np.float64)

model = tf.keras.models.load_model(save_path, custom_objects={"CustomModel": CNN1D})

testLoss, testAcc = model.evaluate(x_test, y_test)
print('\nAccuracy:', testAcc)
print('\nLoss: ', testLoss)

from sklearn.metrics import classification_report, confusion_matrix
yPred = model.predict(x_test)

# convert from one hot encode in string
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



