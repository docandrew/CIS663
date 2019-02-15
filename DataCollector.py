import numpy as np

# get the data
trainFile = 'train.csv'
testFile = 'test.csv'
label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def getTrainData():
    # images are 48x48
    Y = []
    X = []
    first = True
    for line in open(trainFile):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])

    X, Y = np.array(X) / 255.0, np.array(Y)
    N, D = X.shape
    X = X.reshape(N, 48, 48, 1)
    num_class = len(set(Y))
    Y = (np.arange(num_class) == Y[:, None]).astype(np.float32)
    return X, Y, num_class

def getTestData():
    # images are 48x48
    Y = []
    X = []
    first = True
    for line in open(testFile):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])

    X, Y = np.array(X) / 255.0, np.array(Y)
    N, D = X.shape
    X = X.reshape(N, 48, 48, 1)
    num_class = len(set(Y))
    Y = (np.arange(num_class) == Y[:, None]).astype(np.float32)
    return X, Y