
import sys
sys.path.append("..")
import math
import numpy as np
import tensorflow as tf
from Train.FLS_TrainFun import FLS_TrainFun
from Train.FLS_TrainFun_parallel import FLS_TrainFun_parallel
K = 1500
y = np.zeros(K)
G = np.zeros(K)
X_train = np.zeros([1000,2])
Y_train = np.zeros(1000)
X_test = np.zeros([500,2])
Y_test = np.zeros(500)
y[0] = 0
y[1] = 0.05
G[0] = 0

def g(x,y):
    return (x*y*(x+2.5))/(1+x**2+y**2)

def r(x):
    return math.sin(2*math.pi*x/25)+math.sin(math.pi*x/50)+math.sin(math.pi*x/125)

def aux_list(a,b):
    out=[]
    for i in range(b-a):
        out.append(a+i)
    return out 

index = aux_list(2,K)  
#print(index)
for i in index:
    y[i] = 0.6*y[i-1]+0.2*y[i-2]+r(i)

for i in index:
    G[i-1]=g(y[i-1],y[i-2])

for i in range(1000):    
    X_train[i,0] = y[i+1]
    X_train[i,1] = y[i]
    Y_train[i] = G[i+1]

for i in range(500):
    X_test[i,0] = y[i+1]
    X_test[i,1] = y[i]
    Y_test[i] = G[i+1]

#print(X_train)
LL=[['G','G','G','G','G','G'],['G','G','G','G','G','G'],
    ['G','G','G','G','G','G'],['G','G','G','G','G','G'],
    ['G','G','G','G','G','G'],['G','G','G','G','G','G'],
    ['G','G','G','G','G','G'],['G','G','G','G','G','G'],  
    ['G','G','G','G','G','G'],['G','G','G','G','G','G'],
    ['G','G','G','G','G','G'],['G','G','G','G','G','G'],
    ['G','G','G','G','G','G'],['G','G','G','G','G','G'],
    ['G','G','G','G','G','G'],['G','G','G','G','G','G'],
    ['G','G','G','G','G','G'],['G','G','G','G','G','G'],
    ['G','G','G','G','G','G'],['G','G','G','G','G','G'],
    ['G','G','G','G','G','G'],['G','G','G','G','G','G'],
    ['G','G','G','G','G','G'],['G','G','G','G','G','G'],  
    ['G','G','G','G','G','G'],['G','G','G','G','G','G'],
    ['G','G','G','G','G','G'],['G','G','G','G','G','G'],
    ['G','G','G','G','G','G'],['G','G','G','G','G','G'],
    ['G','G','G','G','G','G'],['G','G','G','G','G','G']]


# FLS_TrainFun(24,2,LL,X_train,Y_train,X_test,Ypredict=Y_test,modeName='TSK',modeType=2,predictMode=False,\
#    validationRatio=0.2,XvalidationSet=None,YvalidationSet=None,\
#    optimizer=tf.keras.optimizers.Adam(0.01),lossFunction=tf.keras.losses.mean_squared_error,\
#    batchSIZE=32,epoch=15,useGPU=False,saveMode=False,outputModeName=None,modeSavePath=None)


FLS_TrainFun_parallel(16,2,LL,X_train,Y_train,X_test,Ypredict=Y_test,modeName='Mamdani',modeType=2,predictMode=False,\
    validationRatio=0.1,XvalidationSet=None,YvalidationSet=None,\
    optimizer=tf.keras.optimizers.Adam(0.05),lossFunction=tf.keras.losses.mean_squared_error,\
    batchSIZE=16,epoch=12,subMode_learningRate=tf.constant(0.5),useGPU=False,processesNum=10)   








  
    
    
   