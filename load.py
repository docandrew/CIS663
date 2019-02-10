# Script for renaming the CK+ datasets and just picking the last
# image in the folder.

import os
import shutil

# copy alternating images to training & test folders.

def renameEverything(imageDir, labelDir, outputDir):
    # Get list of subjects
    subjects = []
    for item in os.listdir(imageDir):
        if os.path.isdir(os.path.join(imageDir, item)):
            subjects.append(item)

    labelStrings = ['neutral','anger','contempt','disgust','fear','happy','sadness','surprise']
    labelCounters = [0,0,0,0,0,0,0,0]
    # Get list of labels
    # labelFolders = []
    # for item in os.listdir(labelDir):
    #     if os.path.isdir(os.path.join(labelDir, item)):
    #         labelFolders.append(os.path.join(labelDir, item))
    #print('Label Dir:', labelDir)

    # For each subject, find the emotions
    for subject in subjects:
        subjectFolder = os.path.join(imageDir, subject)

        if(os.path.isdir(subjectFolder)):
            for emotion in os.listdir(subjectFolder):
                emotionFolder = os.path.join(subjectFolder, emotion)

                if(os.path.isdir(emotionFolder)):

                    for image in os.listdir(emotionFolder):
                        imagePath = os.path.join(emotionFolder, image)
                        #print('Checking', imagePath, 'for matching emotion label')
                        folder, imageFilename = os.path.split(imagePath)
                        imageFile, imageExt = os.path.splitext(imageFilename)

                        #matching label will be in
                        # labelDir\subject\emotion\file
                        #print('Label Dir', labelDir)
                        #print('Subject ', subject)
                        #print('Emotion ', emotion)
                        #print('imageFile', imageFile)
                        labelPath = os.path.join(labelDir, subject, emotion, imageFile) + '_emotion.txt'
                        #print('Does', labelPath, 'exist?')
                        if(os.path.exists(labelPath)):
                            #print('Found emotion label')
                            #get emotion label out of file
                            label = int(float(open(labelPath,"r").read()))
                            #print('Emotion is: ', label)

                            #OK, found a label, so copy it to a folder
                            outputPath = os.path.join(outputDir, labelStrings[label]) + '\\' + labelStrings[label] + "{:03}".format(labelCounters[label]) + '.png'
                            labelCounters[label] += 1
                            print('Copying ', imagePath, ' to: ', outputPath)
                            shutil.copyfile(imagePath, outputPath)
    # r = []
    # for root, dirs, files in os.walk(dir):
    #     for subject in dirs:
    #         print(os.path.join(root,subject))
    #         #r.append(os.path.join(root, name))
    # return r

renameEverything('C:\\Users\\doc\\Desktop\\CIS663 Data\\cohn-kanade-images',
                 'C:\\Users\\doc\\Desktop\\CIS663 Data\\cohn-kanade-emotions',
                 'C:\\Users\\doc\\Desktop\\CIS663 Data\\data')