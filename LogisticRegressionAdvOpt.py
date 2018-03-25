import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize
from scipy.optimize import check_grad


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
    costHistory = np.zeros((numIters, 1))
    f1 = computeCost

    #Gradient Descent
    for i in range(numIters):
        # Calculate Cost
        cost = f1(X, y, theta)

        # Get Data Batch
        randIdx = np.random.choice(X.shape[0], batchSize, replace=False)
        batch = X[randIdx, :]
        labels = y[randIdx, :]

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

#FUNCTIONS FOR ADVANCE OPTIMIZATION
def costFunc(theta,X,y):
    #Cost Function
    m = y.shape[0]
    theta = theta.reshape(-1,1)
    y_estimate = sigmoid( X @ theta )
    cost = -y * np.log(y_estimate) - (1 - y) * np.log(1 - y_estimate)
    cost = (1 / m) * sum(cost)

    return cost[0]

def grad(theta,X,y):
    #Calculate Gradient
    y_estimate = sigmoid( X @ theta.reshape(-1,1) )
    gradient = X.T @ (y_estimate - y)

    return gradient.flatten()


def printCost(theta):
    global historyIndex,costH
    currentCost=costFunc(theta,xTrain,yTrain)
    costH[historyIndex] = currentCost
    historyIndex += 1

#GLOBAL VARIABLES
ALPHA = 10
NUM_ITERS = 1500
MINIBATCH_ALPHA= 0.3 * ALPHA
MINIBATCH_SIZE = 50
STOCHASTIC_ALPHA=0.5

maxiters=400
costH =  np.zeros((maxiters,1))
historyIndex = 0

if __name__ == '__main__':

    #Read DataSet
    testSet = pd.read_csv('exerciseData/testSetGenre.csv')
    trainingSet = pd.read_csv('exerciseData/trainingSetGenre.csv')

    #Split X and Y data
    xTrain, yTrain=getValues(trainingSet)
    testX, testY = getValues(testSet)

    #Reshape labels
    yTrain=yTrain.reshape((-1, 1))
    yTrain=yTrain.astype(dtype='float64')
    testY= testY.reshape((-1,1))
    testY=testY.astype(dtype='float64')

    #Feature Scaling
    meanVec, maxVec, minVec = featureScaling(xTrain)

    # Apply Scaling
    xTrain = (xTrain - meanVec) / (maxVec - minVec)
    testX = (testX - meanVec) / (maxVec - minVec)

    #Add bias Column
    xTrain=addBiasColumn(xTrain)
    testX    =addBiasColumn(testX)
    
    # Samples in training set(m), features of each sample(n)
    m = xTrain.shape[0]
    n = xTrain.shape[1]
    print("# of samples in Training Set: {:3d}. # of Features of each sample: {:3d}.\
                        # of Samples in Test Set {:3d}.\n".format(m, n,testY.shape[0]))

    #Initialize Weights
    theta =  np.ones((n,1))
    theta2 = np.ones((n, 1))

    print("Initial Cost", computeCost(xTrain, yTrain, theta))

    #Gradient Descent
    startTime1 = time.time()
    theta,costHistory = gradientDescent(xTrain, yTrain, theta, ALPHA, NUM_ITERS)
    endTime1 = time.time()

    #Advance Minization
    startTime2 = time.time()
    res = minimize(costFunc, theta2, args=(xTrain, yTrain), method="BFGS", \
                   jac=grad, callback=printCost, options={'disp': True, 'maxiter': maxiters})
    endTime2 = time.time()

    print("Gradient Descent Execution time: {:.3f}".format(endTime1 - startTime1))
    print("Advance Optimiza Execution time: {:.3f}".format(endTime2 - startTime2))

    print("Final Cost", computeCost(xTrain, yTrain, theta))

    #Cost Plots during Training
    fig, axes= plt.subplots(1)
    axes.set_title("Logistic Regression Training")
    axes.set_xlabel("Iteration")
    axes.set_ylabel("Cost")

    axes.plot(np.arange(NUM_ITERS), costHistory, linewidth=2.0, \
              label="Exact Gradient, alpha={:.2f}".format(ALPHA))

    axes.plot(np.arange(costH.size), costH, linewidth=2.0, \
              label="Advance Optimization")

    axes.legend()

    #Evaluate model in test set.
    print("Accuracy in test Set (Gradient Descent)",evaluateModel(testX, testY, theta))
    print("Accuracy in test Set (Advance Optimiza)", evaluateModel(testX, testY, res.x.reshape(-1,1)))

    axes.set_xlim(-50, NUM_ITERS)
    axes.grid()
    plt.show()
