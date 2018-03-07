# -*- coding: utf-8 -*-

import numpy as np
from numpy import  matmul
from numpy.linalg import inv
import matplotlib.pyplot as plt


def getData(fileName):
    data=[]
    with open(fileName) as file:
        for line in file:
            data+=[list(map( float, line.strip("\n").strip('\t').split("\t") ))]

    return np.array(data)

def splitData(xData,yData):

    xTrain=xData[:1300,:]
    yTrain=yData[:1300,:]
    xTest=xData[1300:,:]
    yTest=yData[1300:,:]
    return xTrain,yTrain,xTest,yTest

def addBias(data):
    b=np.ones((data.shape[0],data.shape[1]+1))
    b[:,1:]=data
    return b

def computeCost (X,y,theta):
    m = y.shape[0]

    #Cost Function
    y_estimate = X @ theta;
    cost = sum( (y_estimate - y )**2 ) / (2*m);

    return cost[0]

def gradientDescent (X, y, theta, alpha, numIters):

    m = y.shape[0]
    costHistory = np.zeros((numIters,1))
    f1=computeCost

    for i in range(numIters):
        #Calculate Cost
        cost= f1(X,y,theta)

        #Calculate Gradient
        y_estimate = X @ theta;
        gradient = X.transpose() @ ( y_estimate - y)

        #Update parameter vector
        theta = theta - (alpha/m) * gradient

        costHistory[i]=cost

    return theta,costHistory

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
            y_estimate = batch @ theta;
            gradient = batch.transpose() @ (y_estimate - labels)

            # Update parameter vector
            theta = theta - (alpha / batchSize) * gradient

            costHistory[i] = cost

        return theta, costHistory

def stochasticGradientDescent (X, y, theta, alpha, numIters):

    costHistory = np.zeros((numIters,1))
    n=theta.shape[0]
    m=X.shape[0]

    for i in range(numIters):
        cost= computeCost(X,y,theta)

        #calculate stochasticGradient
        randIndex=int(np.random.random()*m)
        data  = X[randIndex,:].reshape(1,n)
        label = y[randIndex,:].reshape(1,1)

        error = (data @ theta - label)
        stochasticGradient = data.transpose() * error

        # Update parameter vector
        theta = theta - (alpha) * stochasticGradient

        costHistory[i]=cost

    return theta,costHistory

xt_file ='xTRAIN10.txt'
yt_file ='yTRAIN10.txt'

alpha = 0.06
numIters=400

if __name__ == "__main__":

    xData=getData(xt_file)
    yData=getData(yt_file)

    xTrain,yTrain,xTest,yTest=splitData(xData,yData)

    #Add bias column
    xTrain=addBias(xTrain)
    xTest=addBias(xTest)

    m=xTrain.shape[0]
    n=xTrain.shape[1]

    #Normal Equation
    bestTheta = inv( xTrain.transpose() @ xTrain ) @ xTrain.transpose()  @ yTrain

    #Initialize Theta
    theta_50   = np.ones((n,1))
    theta_100  = np.ones((n,1))
    theta_200  = np.ones((n,1))
    theta_1000 = np.ones((n,1))

    theta = np.ones((n,1))
    stochasticTheta = np.ones((n,1))
    miniBatchTheta =np.ones((n,1))

    #Stochastic Gradient Descent
    # theta_50,costHistory_50 = stochasticGradientDescent(xTrain[:50,:],yTrain[:50],theta_50,alpha,numIters)
    # theta_100,costHistory_100 = stochasticGradientDescent(xTrain[:100,:],yTrain[:100],theta_100,alpha,numIters)
    # theta_200,costHistory_200 = stochasticGradientDescent(xTrain[:200,:],yTrain[:200],theta_200,alpha,numIters)
    # theta_1000,costHistory_1000 = stochasticGradientDescent(xTrain[:1000,:],yTrain[:1000],theta_1000,alpha,numIters)

    #Exact Gradient
    theta_50,costHistory_50 = gradientDescent(xTrain[:50,:],yTrain[:50],theta_50,alpha,numIters)
    theta_100,costHistory_100 = gradientDescent(xTrain[:100,:],yTrain[:100],theta_100,alpha,numIters)
    theta_200,costHistory_200 = gradientDescent(xTrain[:200,:],yTrain[:200],theta_200,alpha,numIters)
    theta_1000,costHistory_1000 = gradientDescent(xTrain[:1000,:],yTrain[:1000],theta_1000,alpha,numIters)

    #Plots
    fig,axes=plt.subplots(1)
    axes.grid()

    #Plot: CostHistory in Gradient Descent
    axes.set_title('Cost Function Every iteration')
    axes.set_xlabel('Iteration')
    axes.set_ylabel('Cost')
    axes.set_xlim(-20,numIters)

    axes.plot(np.arange(numIters),costHistory_50,linewidth=1.5,label="Training Size: 50")
    axes.plot(np.arange(numIters),costHistory_100,linewidth=1.5,label="Training Size: 100")
    axes.plot(np.arange(numIters),costHistory_200,linewidth=1.5,label="Training Size: 200")
    axes.plot(np.arange(numIters),costHistory_1000,linewidth=1.5,label="Training Size: 1000")

    axes.legend()

    #Measuring Model accuracy
    print("Test Set Cost(50): {:.3f}".format(computeCost(xTest,yTest,theta_50)))
    print("Test Set Cost(100): {:.3f}".format(computeCost(xTest,yTest,theta_100)))
    print("Test Set Cost(200): {:.3f}".format(computeCost(xTest,yTest,theta_200)))
    print("Test Set Cost(1000): {:.3f}".format(computeCost(xTest,yTest,theta_1000)))

    bestTheta50   = inv( xTrain[:50,:].transpose() @ xTrain[:50,:] ) @ xTrain[:50,:].transpose()  @ yTrain[:50]
    bestTheta1000 = inv( xTrain[:1000,:].transpose() @ xTrain[:1000,:] ) @ xTrain[:1000,:].transpose()  @ yTrain[:1000]

    plt.show()