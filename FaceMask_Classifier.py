# importing libraries
import tensorflow as tf
import os
import numpy as np
import pathlib

#---------------------- Path for training and testing dataset---------------------
path = os.getcwd()

train_mask_dir = os.path.join(path+'/data/training/mask')
train_nomask_dir = os.path.join(path+'/data/training/no_mask')

valid_mask_dir = os.path.join(path+'/data/validation/mask')
valid_nomask_dir = os.path.join(path+'/data/validation/no_mask')

train_mask_label = os.listdir(train_mask_dir)
print(train_mask_label)
train_nomask_label = os.listdir(train_nomask_dir)

valid_mask_label = os.listdir(valid_mask_dir)
valid_nomask_label = os.listdir(valid_nomask_dir)

#------------------- Data loaded and augmented on memeory ----------------
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255,
                                                                rotation_range=40,
                                                                width_shift_range=0.2,
                                                                height_shift_range=0.2,
                                                                shear_range=0.2,
                                                                horizontal_flip = True)
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

train_data = train_datagen.flow_from_directory(path+'/data/training',
                                               target_size=(300,300),
                                               batch_size=10,
                                               class_mode='binary'
                                               )

Validation_data = valid_datagen.flow_from_directory(path+'/data/validation',
                                                    target_size=(300,300),
                                                    batch_size=10,
                                                    class_mode='binary'
                                                    )

#--------------- Model Architecture ----------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), input_shape=(300,300,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy']
              )

class mycallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.97):
            print('\n Reached 80% cancelling training')
            self.model.stop_training = True

callback = mycallback()

history = model.fit(
    train_data,
    steps_per_epoch=32,  
    epochs=20,
    batch_size = 64,
    verbose=1,
    validation_data = Validation_data,
    validation_steps=32,
    callbacks=[callback]
    )

#model.save('model.h5')
tf.saved_model.save(model, path)

#------------------- End of training -------------------------------

#------------------ Code below is for testing the model ----------------

model = tf.keras.models.load_model('model.h5')
print('[INFO] Model loaded...')

test_data = tf.keras.preprocessing.image.load_img(path+'/cat.jpg', target_size=(300,300))
test_data = tf.keras.preprocessing.image.img_to_array(test_data)
test_data = np.expand_dims(test_data, axis=0)

test = np.vstack([test_data])
test = test.astype(np.uint8)

prediction = model.predict(test, batch_size=10)
print(prediction[0])
if (prediction[0] < 0.5):
    print('mask detected')
if (prediction[0] > 0.5):
    print('NO mask detected')

# -----------------------createing a TFLite Model -------------------------------
print('[INFO] Starting model convertion')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizarions= [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
print('[INFO] Convertion Succesful...')
print('[INFO] saveing the TFLite model')
tflite_model_file = pathlib.Path('./model.tflite')
tflite_model_file.write_bytes(tflite_model)
print('Done')


interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]
output_details = interpreter.get_output_details()

print(input_index)
print(output_details)
interpreter.set_tensor(input_index, test)
interpreter.invoke()

output_data = interpreter.get_tensor(output_index)
boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
scores = interpreter.get_tensor(output_details[2]['index'])[0] 
print(output_data[:])
print(boxes)
print(classes[:2])
print(scores[:2])
