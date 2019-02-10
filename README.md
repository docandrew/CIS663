# CIS663

First attempt: Using 48x48 pixel images, 80/20 training/validation split
* 1549777156.7452652.json
* 1549777156.7452652.h5

Second attempt: Using 128x128 pixel images, 80/20 training/validation split
* '128px 1-channel color (grayscale).h5'
* '128px 1-channel color (grayscale).json'

# Files:
load.py - used to take the CK+ dataset and sort into folders by emotion,
so the Keras ImageDataGenerator can set appropriate labels.

preprocess.py - Convert images to grayscale, then use a Haar Cascade
Classifier used to identify the "face" part of each image, crop the
face and resize.

tfemotion.py - Classifier, using Keras. Will save model JSON and the
model weights in h5 format after a training & validation run.

applymodel.py - Webcam application to take a frame of video, detect
faces, crop and resize (the same as preprocess.py) and then use
the model output from tfemotion.py to see if it can classify an
emotion.