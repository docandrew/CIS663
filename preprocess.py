import cv2
import os

def cropFace(filePath, haarFaceCascade):
    # filePath = os.path.join(rootPath, 'data\\sadness\\sadness004.png')
    print('Loading ', filePath)
    testImage = cv2.imread(str(filePath))
    grayImage = cv2.cvtColor(testImage, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = haarFaceCascade.detectMultiScale(grayImage)
    print('Found',len(faces), 'faces')

    if(len(faces) == 1):
        (x, y, w, h) = faces[0]
        # Plot rectangle
        #cv2.rectangle(grayImage, (x, y), (x+w, y+w), (255, 255, 255), 2)

        #crop image
        croppedImage = grayImage[y:y+h, x:x+w].copy()
        croppedImage = cv2.resize(croppedImage, (48, 48))
        return croppedImage
    else:
        print('Unable to find face in ', filePath)
        return None

        # cv2.imshow('Test Image Cropped', croppedImage)
        # cv2.imshow('Test Image Face', grayImage)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

def cropImages(rootPath, haarFaceCascade):
    # load face
    #for root, dirs, files in os.walk(os.path.join(rootPath, 'data')):
    #    for name in files:
    for emotion in os.listdir(os.path.join(rootPath, 'data')):
        for file in os.listdir(os.path.join(rootPath, 'data', emotion)):
            #filePath = os.path.join(rootPath, 'data', emotion)
            #fileName = os.path.split(filePath)
            fileName = os.path.split(file)[1]
            print('emotion: ', emotion)
            print('file: ', file)
            filePath = str(os.path.join(rootPath, 'data', str(emotion), str(fileName)))
            outputPath = str(os.path.join(rootPath, 'croppedData', str(emotion), str(fileName)))

            faceImage = cropFace(filePath, haarFaceCascade)
            if faceImage is not None:
                print('Cropping face from ', filePath, 'to', outputPath)
                cv2.imwrite(outputPath, faceImage)

#print('Using OpenCV Version: ', cv2.__version__)

# load Haar cascade classifier
rootPath = 'C:\\Users\\doc\\Desktop\\CIS663 Data\\'
haarPath = os.path.join(rootPath, 'HaarCascadeClassifier', 'haarcascade_frontalface_alt.xml')
print('Using Haar Cascade Classifier: ', haarPath)
haarFaceCascade = cv2.CascadeClassifier(haarPath)
cropImages(rootPath, haarFaceCascade)