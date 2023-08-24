from tensorflow.keras.models import load_model

# Loading the Previous Model
pretrain_model = load_model('pretrain_model.h5')

# Setting Semi-Deep Fine-Tuning
pretrain_model.layers[6].trainable = True
pretrain_model.layers[7].trainable = True

# Learning
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
pretrain_model.compile(optimizer=optimizer, loss=cost_function, metrics=['accuracy'])

# Semi-Deep Fine-Tuning
epochs = 4
before_fine_tuning = pretrain_model.evaluate(test_data, test_labels)  # Accuracy before Fine-Tuning
for epoch in range(epochs):
    pretrain_model.fit(train_data, train_labels, epochs=1, batch_size=batch_size, validation_data=(val_data, val_labels))
    acc_fine_tuning = pretrain_model.evaluate(test_data, test_labels)[1]  # Accuracy after Fine-Tuning
    if acc_fine_tuning > before_fine_tuning:
        print("Train Loss: {}, Validation Loss: {}, Train Accuracy: {}, Validation Accuracy: {}".format(
            pretrain_model.history.history['loss'], pretrain_model.history.history['val_loss'],
            pretrain_model.history.history['accuracy'], pretrain_model.history.history['val_accuracy']
        ))
        break