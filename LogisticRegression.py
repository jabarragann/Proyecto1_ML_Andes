import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time


def createDataSet():
    data = pd.read_csv('exerciseData/msd_genre_dataset.csv')

    metalPunkData = data[(data.genre == 'metal') | (data.genre == 'punk') ]
    metalPunkData.to_csv("exerciseData/Metal_Punk.csv")

    danceElectronicaData = data[(data.genre == 'dance and electronica')]
    danceElectronicaData.to_csv("exerciseData/Dance_Electronica.csv")

    # Remove Track_id, artist_name, title
    print("Dance data size:",danceElectronicaData.shape[0])
    print("Metal & Punk data size:",metalPunkData.shape[0])

    headers = list(danceElectronicaData.columns.values)

    danceElectronicaData = danceElectronicaData.drop(labels=[headers[1],headers[2],headers[3]],axis=1)
    metalPunkData = metalPunkData.drop(labels=[headers[1],headers[2],headers[3]],axis=1)

    #Create Training Set and Test set
    df1 = danceElectronicaData.sample(frac=0.2, replace=False)
    danceElectronicaData = danceElectronicaData.drop(df1.index.values)

    df2 = metalPunkData.sample(frac=0.2, replace=False)
    metalPunkData = metalPunkData.drop(df2.index.values)

    testSet = pd.concat([df1, df2])
    trainingSet = pd.concat([danceElectronicaData,metalPunkData])

    testSet = testSet.sample(frac=1, replace=False)
    testSet.index.name='data_base_index'
    trainingSet = trainingSet.sample(frac=1, replace=False)
    trainingSet.index.name="data_base_index"

    print("Training set size:", trainingSet.shape[0])
    print("Test set size:", testSet.shape[0])

    # Save Training Set And Test Set
    trainingSet.to_csv("exerciseData/trainingSetGenre.csv")
    testSet.to_csv("exerciseData/testSetGenre.csv")

def getValues(dataSet):


    yData = dataSet['genre'].values
    xData = dataSet.drop(labels=['genre', "data_base_index"], axis=1).values

    for i in range(yData.shape[0]):
        if yData[i] == "dance and electronica":
            yData[i] = 1
        else:
            yData[i] = 0

    return xData,yData

def featureScaling(X):

    meanVec=np.mean(X,axis=0)
    maxVec=X.max(axis=0)
    minVec=X.min(axis=0)

    return meanVec,maxVec,minVec

def addBiasColumn(X):
    temp=X
    m = X.shape[0]
    n = X.shape[1]

    X = np.ones((m,n+1))
    X[:,1:]=temp

    return X

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def computeCost(X,y,theta):


    #Size
    m=y.shape[0]
    #Estimates
    z = X @ theta
    y_estimate = sigmoid(z)
    #Cost Function
    cost = -y* np.log(y_estimate) -(1-y)*np.log(1-y_estimate)

    return (1/m)*sum(cost)

def gradientDescent(X, y, theta, alpha, numIters):
    m = y.shape[0]
    costHistory = np.zeros((numIters, 1))
    f1 = computeCost

    for i in range(numIters):
        # Calculate Cost
        cost = f1(X, y, theta)

        # Calculate Gradient
        y_estimate =sigmoid( X @ theta );
        gradient = X.transpose() @ (y_estimate - y)

        # Update parameter vector
        theta = theta - (alpha / m) * gradient

        costHistory[i] = cost

    return theta, costHistory


def stochasticGradientDescent(X, y, theta, alpha, numIters):
    costHistory = np.zeros((numIters, 1))
    n = theta.shape[0]
    m = X.shape[0]

    for i in range(numIters):
        cost = computeCost(X, y, theta)

        # calculate stochasticGradient
        randIndex = int(np.random.random() * m)
        data = X[randIndex, :].reshape(1, n)
        label = y[randIndex, :].reshape(1, 1)

        error = (sigmoid(data @ theta) - label)
        stochasticGradient = data.transpose() * error

        # Update parameter vector
        theta = theta - (alpha) * stochasticGradient

        costHistory[i] = cost

    return theta, costHistory

def minibatchGradientDescent(X,y,theta,alpha,numIters, batchSize):
    m = y.shape[0]
    n = X.shape[1]
    costHistory = np.zeros((numIters, 1))
    f1 = computeCost

    for i in range(numIters):

        batch  = np.ones((batchSize,n))
        labels = np.ones((batchSize, 1))

        #Gradient Descent
        for i in range(numIters):
            # Calculate Cost
            cost = f1(X, y, theta)

            #Get Data Batch
            usedIndex = np.ones((batchSize, 1)) * (-1)
            for j in range(batchSize):
                randIndex = int(np.random.random() * m)
                while randIndex in usedIndex:  randIndex = int(np.random.random() * m)

                usedIndex[j] = randIndex

                batch[j, :] = X[randIndex, :].reshape(1, n)
                labels[j, :] = y[randIndex, :].reshape(1, 1)

            # Calculate Gradient
            y_estimate = sigmoid(batch @ theta);
            gradient = batch.transpose() @ (y_estimate - labels)

            # Update parameter vector
            theta = theta - (alpha / batchSize) * gradient

            costHistory[i] = cost

        return theta, costHistory

def evaluateModel(X,y,theta):

    #Size
    m=y.shape[0]
    #Estimates
    z = X @ theta
    y_estimate = sigmoid(z)

    for i in range(m):
        if y_estimate[i] > 0.5:
            y_estimate[i]=1
        else:
            y_estimate[i]=0

    temp=np.equal(y_estimate,y)
    count=0
    for i in temp:
        if i:
            count+=1

    #Cost Function
    accuracy = count/m
    return accuracy

alpha = 10
numIters = 2000

minibatchAlpha= 0.3*alpha
minibatchSize = 50

stochasticAlpha=0.5

if __name__ == '__main__':

    #createDataSet()

    #Read DataSet
    testSet = pd.read_csv('exerciseData/testSetGenre.csv')
    trainingSet = pd.read_csv('exerciseData/trainingSetGenre.csv')

    #Split X and Y data
    trainingX,trainingY=getValues(trainingSet)
    testX, testY = getValues(testSet)

    #m = # of samples in data set
    #n = # of features
    m=trainingX.shape[0]
    n=trainingX.shape[1]

    #Reshape labels
    trainingY=trainingY.reshape((m,1))
    trainingY=trainingY.astype(dtype='float64')
    testY= testY.reshape((2048,1))
    testY=testY.astype(dtype='float64')

    #Feature Scaling
    meanVec, maxVec, minVec = featureScaling(trainingX)

    # Apply Scaling
    trainingX = (trainingX - meanVec) / (maxVec - minVec)
    testX = (testX - meanVec) / (maxVec - minVec)

    #Add bias Column
    trainingX=addBiasColumn(trainingX)
    testX    =addBiasColumn(testX)

    #Initialize Weights
    theta = np.ones((n+1,1))
    stochasticTheta = np.ones((n+1,1))
    minibatchTheta = np.ones((n+1,1))

    print("Initial Cost", computeCost(trainingX,trainingY,theta))

    #Apply Gradient Descent
    startTime = time.time()
    theta,costHistory = gradientDescent(trainingX,trainingY, theta, alpha, numIters)
    endTime = time.time()
    print("Gradient Descent Execution time: {:.3f}".format(endTime - startTime))

    #Apply minibatch Gradient Descent
    startTime = time.time()
    minibatchTheta,minibatchCost = minibatchGradientDescent(trainingX,trainingY,minibatchTheta,alpha*0.8,numIters,50)
    endTime = time.time()
    print("MiniBatch Gradient Descent Execution time: {:.3f}".format(endTime - startTime))

    #Apply Stochastic Gradient Descent
    startTime = time.time()
    stochasticTheta, stochasticCostHistory = stochasticGradientDescent(trainingX,trainingY, stochasticTheta, stochasticAlpha, numIters)
    endTime = time.time()
    print("Stochastic Gradient Descent Execution time: {:.3f}".format(endTime - startTime))

    print("Final Cost", computeCost(trainingX, trainingY, theta))

    #Cost Plots during Training
    fig, axes= plt.subplots(1)
    axes.set_title("Logistic Regression Training")
    axes.set_xlabel("Iteration")
    axes.set_ylabel("Cost")

    axes.plot(np.arange(numIters), costHistory,linewidth=2.0,\
                        label="Exact Gradient, alpha={:.2f}".format(alpha))

    axes.plot(np.arange(numIters),minibatchCost,linewidth=1.5,linestyle='dashed',\
                        label="Minibatch({:d}) Gradient Descent,alpha={:.2f}".format(minibatchSize,minibatchAlpha))

    axes.plot(np.arange(numIters), stochasticCostHistory,linewidth=1.5,linestyle='dashed', \
              label="Stochastic Descent,alpha={:.2f}".format(stochasticAlpha))

    axes.legend()

    #Evaluate model in test set.
    print("Accuracy in training Set", evaluateModel(trainingX, trainingY, theta))
    print("Accuracy in test Set (Exact Gradient)",evaluateModel(testX, testY, theta))
    print("Accuracy in test Set (Minibatch Gradient)", evaluateModel(testX, testY, minibatchTheta))
    print("Accuracy in test Set (Stochastic Gradient)", evaluateModel(testX, testY, stochasticTheta))

    axes.set_xlim(-50,numIters)
    axes.grid()
    plt.show()
