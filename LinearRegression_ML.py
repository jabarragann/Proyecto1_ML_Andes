# -*- coding: utf-8 -*-

import numpy as np 
from numpy import  matmul
from numpy.linalg import inv
import matplotlib.pyplot as plt
import time




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

xt_file ='exerciseData/xTRAIN10.txt'
yt_file ='exerciseData/yTRAIN10.txt'

alpha = 0.06 
numIters=240
miniBatchSize=3
    
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
    theta = np.ones((n,1))    
    stochasticTheta = np.ones((n,1))
    miniBatchTheta =np.ones((n,1))


    # Gradient Descent
    startTime = time.time()
    theta,costHistory = gradientDescent(xTrain,yTrain,theta,alpha,numIters)
    endTime = time.time()
    print("Gradient Descent Execution time: {:.3f}".format(endTime-startTime))

    #Stochastic Gradient Descent
    startTime = time.time()
    stochasticTheta,stochasticCostHistory = stochasticGradientDescent(xTrain,yTrain,stochasticTheta,alpha,numIters)
    endTime = time.time()
    print("Stochastic Gradient Descent Execution time: {:.3f}".format(endTime - startTime))

    #miniBatch Gradient Descent
    startTime = time.time()
    miniBatchTheta,miniBatchCost= minibatchGradientDescent(xTrain,yTrain,miniBatchTheta,alpha,numIters,miniBatchSize)
    endTime = time.time()
    print("Minibatch Gradient Descent Execution time: {:.3f}".format(endTime - startTime))

    #Plots 
    fig,axes=plt.subplots(1)
    axes.grid()

    #Plot: CostHistory in Gradient Descent
    axes.set_title('Cost Function Every iteration')
    axes.set_xlabel('Iteration')
    axes.set_ylabel('Cost')
    axes.set_xlim(-20,numIters)
    axes.plot(np.arange(numIters),costHistory,linewidth=2.0,label="Exact Gradient Calculation,alpha={:.3f}".format(alpha))
    axes.plot(np.arange(numIters),stochasticCostHistory,linewidth=2.0,linestyle='dashed',label="Stochastic Gradient Descent,alpha={:.3f}".format(alpha))
    axes.plot(np.arange(numIters), miniBatchCost,linewidth=2.0,linestyle='dashed',label="Minibatch Gradient Descent({:d}),alpha={:.3f}".format(miniBatchSize,alpha))
    axes.legend()
    
    #Plot: CostHistory in stochastic Gradient Descent
    #Test set accuracy
    print("Test LSM(exact Gradient): {:.4f}".format(computeCost(xTest,yTest,theta)) )
    print("Test LSM(exact minibatch Gradient): {:.4f}".format(computeCost(xTest, yTest, miniBatchTheta)))
    print("Test LSM(exact Stochastic Gradient): {:.4f}".format(computeCost(xTest, yTest, stochasticTheta)))


    ##Evaluate unseen data
    newData = getData("exerciseData/xTEST10.txt")

    result =  xTest @ theta

    resultFile=open("exerciseData/xTEST10Result.txt",'w')
    for i in result:
        resultFile.write( str(i[0])+"\n")


    plt.show()
