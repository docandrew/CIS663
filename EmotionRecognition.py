import DataCollector as dataHelper
import numpy as np
from keras.models import Sequential
from keras.layers import Dense , Activation , Dropout ,Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.metrics import categorical_accuracy
from keras.models import model_from_json

from keras.optimizers import *
from keras.layers.normalization import BatchNormalization

#--------------------------------------------------------
#import tensorflow as tf
#work around allocation error
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#session = tf.Session(config=config)
#--------------------------------------------------------

batch_size = 128
epochs = 30

X_Train, Y_Train, num_class = dataHelper.getTrainData()
X_Test, Y_Test = dataHelper.getTestData()

#Main CNN model with four Convolution layer & two fully connected layer
def baseline_model():
    # Initialising the CNN
    model = Sequential()

    # 1 - Convolution
    model.add(Conv2D(64,(3,3), padding='same', input_shape=(48, 48,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.30))

    # 2nd Convolution layer
    model.add(Conv2D(128,(5,5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.30))

    # 3rd Convolution layer
    model.add(Conv2D(512,(3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.30))

    # 4th Convolution layer
    model.add(Conv2D(512,(3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.30))

    # 5th Convolution layer
    #model.add(Conv2D(512,(3,3), padding='same'))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.30))

    # Flattening
    model.add(Flatten())

    # Fully connected layer 1st layer
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.30))

    # Fully connected layer 2nd layer
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.30))

    # Fully connected layer 3rd layer
    #model.add(Dense(512))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    #model.add(Dropout(0.30))

    model.add(Dense(num_class, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[categorical_accuracy])
    return model


def baseline_model_saved():
    #load json and create model
    json_file = open('model_4layer_2_2_pool.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    #load weights from h5 file
    model.load_weights("model_4layer_2_2_pool.h5")

    adam = optimizers.Adam(lr=0.01, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, decay=0.0, amsgrad = False)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=[categorical_accuracy])

    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[categorical_accuracy])
    return model

is_model_saved = False

# If model is not saved train the CNN model otherwise just load the weights
if(is_model_saved==False ):
    # Train model
    model = baseline_model()
    # Note : 3259 samples is used as validation data &   28,709  as training samples

    model.fit(X_Train, Y_Train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_split=0.1111)
    model_json = model.to_json()
    with open("model_4layer_2_2_pool.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_4layer_2_2_pool.h5")
    print("Saved model to disk")
else:
    # Load the trained model
    print("Load model from disk")
    model = baseline_model_saved()


# Model will predict the probability values for 7 labels for a test image
score = model.predict(X_Test)
print (model.summary())

new_X = [np.argmax(item) for item in score ]
y_test2 = [np.argmax(item) for item in Y_Test]

# Calculating categorical accuracy taking label having highest probability
accuracy = [(x==y) for x,y in zip(new_X,y_test2)]
print(" Accuracy on Test set : " , np.mean(accuracy))
