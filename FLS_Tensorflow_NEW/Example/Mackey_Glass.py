
import sys

from numpy.core.numeric import False_
sys.path.append("..")
import math
import numpy as np
import tensorflow as tf
from Train.FLS_TrainFun import FLS_TrainFun
from Train.FLS_TrainFun_parallel import FLS_TrainFun_parallel
from Train.FLS_TrainFun_parallel_1 import FLS_TrainFun_parallel_1

def DF(x):
    a=0.2
    return (a*x)/(1+x**10)

def Mackey_Glass(N,tau):
    t=np.zeros(N)
    x=np.zeros(N)
    x[0],t[0]=1.2,0
    b,h=0.1,0.1
    for k in range(N-1):
        t[k+1]=t[k]+h
        if t[k]<tau:
            k1=-b*x[k]
            k2=-b*(x[k]+h*k1/2)
            k3=-b*(x[k]+k2*h/2)
            k4=-b*(x[k]+k3*h)
            x[k+1]=x[k]+(k1+k2*2+2*k3+k4)*h/6
        else:
            n=math.floor((t[k]-tau-t[0])/h+1)
            k1=DF(x[n])-b*x[k]
            k2=DF(x[n])-b*(x[k]+h*k1/2)
            k3=DF(x[n])-b*(x[k]+k2*h/2)
            k4=DF(x[n])-b*(x[k]+k3*h)
            x[k+1]=x[k]+(k1+2*k2+2*k3+k4)*h/6
    return x,t

tao=31
N=40020
n_train=4000
y,_=Mackey_Glass(N,tao)
x_star=np.zeros(n_train)
for i in range(n_train):
    x_star[i]=y[i+10]
#plt.plot(arange(1,n_train+1,1),x_star)
#plt.show()



AntecedentsNum=4
data_size=500
multiple=1
X_train=np.zeros([multiple*data_size-3,AntecedentsNum])
Y_train=np.zeros(multiple*data_size-3)
X_test=np.zeros([data_size-3,AntecedentsNum])
Y_test=np.zeros(data_size-3)

X_train[:,0]=x_star[0:multiple*data_size-3]
X_train[:,1]=x_star[1:multiple*data_size-2]
X_train[:,2]=x_star[2:multiple*data_size-1]
X_train[:,3]=x_star[3:multiple*data_size]
Y_train=x_star[4:multiple*data_size+1]

X_test[:,0]=x_star[multiple*data_size+1:(multiple+1)*data_size-2]
X_test[:,1]=x_star[multiple*data_size+2:(multiple+1)*data_size-1]
X_test[:,2]=x_star[multiple*data_size+3:(multiple+1)*data_size]
X_test[:,3]=x_star[multiple*data_size+4:(multiple+1)*data_size+1]
Y_test=x_star[multiple*data_size+5:(multiple+1)*data_size+2]


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

# FLS_TrainFun(16,4,LL,X_train,Y_train,X_test,Ypredict=Y_test,modeName='FWA',modeType=2,predictMode=False,\
#     validationRatio=0.2,XvalidationSet=None,YvalidationSet=None,\
#     optimizer=tf.keras.optimizers.Adam(0.01),lossFunction=tf.keras.losses.mean_squared_error,\
#     batchSIZE=32,epoch=5,useGPU=False,saveMode=False,outputModeName=None,modeSavePath=None)


# FLS_TrainFun_parallel(16,4,LL,X_train,Y_train,X_test,Ypredict=Y_test,modeName='Mamdani',modeType=2,predictMode=False,\
#     validationRatio=0.2,XvalidationSet=None,YvalidationSet=None,\
#     optimizer=tf.keras.optimizers.Adam(0.1),lossFunction=tf.keras.losses.mean_squared_error,\
#     batchSIZE=32,epoch=15,useGPU=True,processesNum=12)   

FLS_TrainFun_parallel_1(16,4,LL,X_train,Y_train,X_test,Ypredict=Y_test,modeName='Mamdani',modeType=2,predictMode=False,\
    validationRatio=0.1,XvalidationSet=None,YvalidationSet=None,\
    optimizer=tf.keras.optimizers.Adam(0.005),lossFunction=tf.keras.losses.mean_squared_error,\
    batchSIZE=32,epoch=10,subMode_learningRate=tf.constant(0.5),useGPU=True,processesNum=12)     