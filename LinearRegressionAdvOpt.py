# -*- coding: utf-8 -*-

import numpy as np 
from numpy.linalg import inv
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize
from scipy.optimize import check_grad


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
        y_estimate = X.dot(theta);
        gradient = X.T.dot( y_estimate - y)
        
        #Update parameter vector
        theta = theta - (alpha/m) * gradient
        
        costHistory[i]=cost
    
    return theta,costHistory

def minibatchGradientDescent(X,y,theta,alpha,numIters, batchSize=4):
    m = y.shape[0]
    n = X.shape[1]
    costHistory = np.zeros((numIters, 1))
    f1 = computeCost

    #Gradient Descent
    for i in range(numIters):
        # Calculate Cost
        cost = f1(X, y, theta)

        #Get Data Batch
        randIdx = np.random.choice(X.shape[0], batchSize, replace=False)
        batch   = X[randIdx,:]
        labels  = y[randIdx,:]

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

def costFunc(theta,X,y): 
    #Cost Function
    m = y.shape[0]       
    y_estimate = X.dot(theta.reshape(-1,1));
    cost = sum( (y_estimate - y )**2 ) / (2*m);
    return cost[0]

def grad(theta,X,y): 
    #Calculate Gradient
    y_estimate = X @ theta.reshape(-1,1);
    gradient = X.T @ (y_estimate - y)

    return gradient.flatten()

def aproxGrad(theta,X,y):
    #Calculate aproximated Gradient with a Batch
    randIdx = np.random.choice(X.shape[0], 5, replace=False)
    X = X[randIdx, :]
    y = y[randIdx, :]
    y_estimate = X.dot(theta.reshape(-1, 1));
    gradient = X.T.dot(y_estimate - y)

    return gradient.flatten()

def printCost(theta):
    global historyIndex,costH
    currentCost=costFunc(theta,xTrain,yTrain)
    costH[historyIndex] = currentCost
    historyIndex += 1

#GLOBAL VARIABLES
xt_file ='exerciseData/xTRAIN10.txt'
yt_file ='exerciseData/yTRAIN10.txt'

alpha = 0.06 
numIters=240
miniBatchSize=3
maxiters=400
costH =  np.zeros((maxiters,1))
historyIndex = 0
    
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
    theta  = np.ones((n,1))
    theta2 = np.ones((n,1))
    
    # Gradient Descent
    startTime1 = time.time()
    theta,costHistory = minibatchGradientDescent(xTrain,yTrain,theta,alpha,numIters)
    endTime1 = time.time()

    startTime2 = time.time()
    res = minimize(costFunc, theta2,args=(xTrain,yTrain), method="CG", \
                   jac=grad, callback=printCost,  options={'disp': True,'maxiter':maxiters})
    endTime2 = time.time()

    print("\nGradient Descent Execution time: {:.3f}".format(endTime1 - startTime1))
    print("Advance Minimization Execution time: {:.3f}".format(endTime2 - startTime2))

    
    #Plots 
    fig,axes=plt.subplots(1)
    axes.grid()

    #Plot: CostHistory in Gradient Descent
    axes.set_title('Cost Function Every iteration')
    axes.set_xlabel('Iteration')
    axes.set_ylabel('Cost')
    axes.set_xlim(-20,numIters)
    axes.plot(np.arange(numIters),costHistory,linewidth=2.0,label="Exact Gradient Calculation,alpha={:.3f}".format(alpha))
    axes.legend()
    
    #Test set accuracy
    print("Test LSM(Gradient Descent): {:.4f}".format(computeCost(xTest,yTest,theta)) )
    print("Test LSM(Advance Optimiza): {:.4f}".format(computeCost(xTest,yTest,res.x.reshape(-1,1) )) )

    #print(costH[:16])

    plt.show()
