import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from time import time
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K

# work around allocation error
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

dataPath = 'C:\\users\\Doc\\Desktop\\CIS663 Data\\croppedData128\\'
img_width, img_height = 128, 128
epochs = 50
batch_size = 16

# to add mutations of the data for more training
# datagen = ImageDataGenerator(rotation_range=40,
#                              width_shift_range=0.2,
#                              height_shift_range=0.2,
#                              shear_range=0.2,
#                              zoom_range=0.2,
#                              horizontal_flip=True,
#                              fill_mode='nearest')
datagen = ImageDataGenerator(validation_split=0.2, rescale=1. / 255)

train_generator = datagen.flow_from_directory(directory=dataPath, color_mode='grayscale', target_size=(img_width,img_height), batch_size=batch_size, subset='training')

validation_generator = datagen.flow_from_directory(directory=dataPath, color_mode='grayscale', target_size=(img_width,img_height), batch_size=batch_size, subset='validation')

input_shape = (img_width, img_height, 1)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(8))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
                metrics=['accuracy'])

print(model.summary())

#logName = time()
logName = '128px 1-channel color (grayscale)'
tensorboard = TensorBoard(log_dir="logs/{}".format(logName), write_graph=True, write_images=True, histogram_freq=1)

model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[tensorboard])

print('Saving model to .json file')
model_json = model.to_json()
with open('{}.json'.format(logName), 'w') as json_file:
    json_file.write(model_json)
print('Saving model weights to .h5 file')
model.save_weights('{}.h5'.format(logName))