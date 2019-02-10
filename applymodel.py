from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
import tensorflow as tf
import numpy
import os
import cv2
import operator

# work around allocation error
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# load json and create model
filePathRoot = 'I:\\Dropbox\\Syracuse\\CIS 663 - Biometrics\\Project\\localrepo\\EmotionEngine\\1549777156.7452652'
json_file = open(filePathRoot + '.json')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into model
loaded_model.load_weights(filePathRoot + '.h5')

# evaluate loaded model on new data
loaded_model.compile(loss='binary_crossentropy',
                    optimizer='rmsprop',
                    metrics=['accuracy'])

# load Haar cascade classifier
rootPath = 'C:\\Users\\doc\\Desktop\\CIS663 Data\\'
haarPath = os.path.join(rootPath, 'HaarCascadeClassifier', 'haarcascade_frontalface_alt.xml')
print('Using Haar Cascade Classifier: ', haarPath)
haarFaceCascade = cv2.CascadeClassifier(haarPath)

# get screengrab from webcam
video_capture = cv2.VideoCapture(0)

currEmotion = 'neutral'
labelStrings = ['anger','contempt','disgust','fear','happy','neutral','sadness','surprise']

while True:
    ret, frame = video_capture.read()

    if frame is None:
        #print('NO frame')
        continue

    #print('Got frame')
    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = haarFaceCascade.detectMultiScale(grayImage)

    cv2.putText(frame, currEmotion, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if (len(faces) == 1):
        (x, y, w, h) = faces[0]
        # Plot rectangle
        # TODO: perform smoothing on this so it's less jittery
        cv2.rectangle(frame, (x, y), (x+w, y+w), (0, 255, 0), 2)

        # crop image
        croppedImage = grayImage[y:y + h, x:x + w].copy()
        croppedImage = cv2.resize(croppedImage, (48, 48))
        #put color back in for classifier (TODO: make this work on grayscale all the way)
        croppedImage = cv2.cvtColor(croppedImage, cv2.COLOR_GRAY2BGR)
        x = img_to_array(croppedImage)      # numpy array with shape (3, 48, 48)
        x = x.reshape((1,) + x.shape)       # numpy array with shape (1, 3, 48, 48)
        predictedEmotion = loaded_model.predict(x)[0]

        print(predictedEmotion)
        index, value = max(enumerate(predictedEmotion), key=operator.itemgetter(1))

        print('Highest Index: ', index)
        if predictedEmotion[int(index)] == 0:
            currEmotion = 'neutral'
        else:
            currEmotion = labelStrings[int(index)]

    cv2.imshow('EmotionEngine - press Q to quit', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print('Turning off Webcam.')
video_capture.release()
cv2.destroyAllWindows()
print('Done.')