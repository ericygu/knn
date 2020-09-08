import argparse
import numpy as np
import pandas as pd
from collections import Counter

def euclidean_distance(element1, element2):
    distance = 0
    for i in range(len(element1)):
        distance += (element1[i] - element2[i]) ** 2
    return distance ** .5

class Knn(object):
    k = 0    #number of neighbors to use
    matrix = None
    array = None

    def __init__(self, k):
        self.k = k

    def train(self, xFeat, y):
        self.matrix = xFeat
        self.array = y
        return self

    def predict(self, xFeat):    
        myHat = [self.iterative_predict(i) for i in xFeat.to_numpy()]
        return myHat
        
    def iterative_predict(self, x):     
        differences = np.array([euclidean_distance(x, y) for y in self.matrix.to_numpy()])
        samples = np.argsort(differences)
        nearest_labels = [self.array[i] for i in samples[:self.k]]
        max_common = Counter(nearest_labels).most_common(1)
        return max_common[0][0]
    
def accuracy(yHat, yTrue):
    acc = np.sum(np.array(yHat) == np.array(yTrue))/len(yHat)
    return acc


def main():
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("k",
                        type=int,
                        help="the number of neighbors")
    parser.add_argument("--xTrain",
                        default="q3xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q3yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q3xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q3yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)
    # create an instance of the model
    knn = Knn(args.k)
    knn.train(xTrain, yTrain['label'])
    # predict the training dataset
    yHatTrain = knn.predict(xTrain)
    trainAcc = accuracy(yHatTrain, yTrain['label'])
    # predict the test dataset
    yHatTest = knn.predict(xTest)
    testAcc = accuracy(yHatTest, yTest['label'])
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)


if __name__ == "__main__":
    main()
