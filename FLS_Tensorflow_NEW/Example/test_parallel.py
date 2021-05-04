#/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/5/4
# @Author  : Wangjiawen

# Note : Parallel algorithm must be run on Linux system.


import sys
sys.path.append("..")
import math
import numpy as np
import os
import time
import random
import tensorflow as tf

from multiprocessing import Process, Queue       #使用多进程加速，实现并行算法

from tensorflow.python.eager.backprop import GradientTape
from tensorflow.python.ops.functional_ops import Gradient
#导入2型FLS系统
from ST2FLS.SingleT2FLS_Mamdani import *
from ST2FLS.SingleT2FLS_TSK import *
from ST2FLS.SingleT2FLS_FWA import *
#导入1型FLS系统
from ST1FLS.SingleT1FLS_Mamdani import *
from ST1FLS.SingleT1FLS_TSK import *


# 子进程调用
def SubMode_train(MODE,lossFunction,Xtrain_subMode,Ytrain_subMode,batch_size,queue,learn_rate=tf.constant(0.001)):
    lackNum = batch_size-len(Xtrain_subMode) % batch_size
    copy_sample_id = random.sample(range(0,len(Xtrain_subMode)),lackNum)
    Xtrain_subMode = np.r_[Xtrain_subMode,Xtrain_subMode[copy_sample_id,:]]
    Ytrain_subMode = np.r_[Ytrain_subMode,Ytrain_subMode[copy_sample_id]]

    subMode_grade = []
    count = 0
    for g in MODE.trainable_variables:
        count += 1
        subMode_grade.append(tf.zeros(g.shape))
    
    for Block_id in range(len(Xtrain_subMode) // batch_size):
        with tf.GradientTape() as tape:
            output_data=MODE(Xtrain_subMode[Block_id*batch_size:(Block_id+1)*batch_size,:])
            Loss=lossFunction(output_data,Ytrain_subMode[Block_id*batch_size:(Block_id+1)*batch_size])
        grades=tape.gradient(Loss,MODE.trainable_variables)
        # for i in range(len(grades)):
        #     MODE.trainable_variables[i] = MODE.trainable_variables[i] + learn_rate*grades[i]

        tf.keras.optimizers.Adagrad(learn_rate).apply_gradients(zip(grades,MODE.trainable_variables))
        print('>>>Processes id:{}, Block_id:{}/{},Block_loss:{}'.format(os.getpid(),Block_id+1,len(Xtrain_subMode)//batch_size,Loss))
        for g_id in range(count):
             subMode_grade[g_id] += grades[g_id]
    #queue.put((subMode_grade,Loss))
    queue.put((MODE.trainable_variables,Loss,subMode_grade))


def FLS_TrainFun_parallel_1(Rule_num,Antecedents_num,InitialSetup_List,Xtrain,Ytrain,Xpredict,Ypredict=None,\
    modeName='Mamdani',modeType=2,predictMode=True,optimizer=tf.keras.optimizers.Adam(0.05),\
    lossFunction=tf.keras.losses.mean_squared_error,batchSIZE=1,epoch=5,subMode_learningRate=tf.constant(0.01),processesNum=None):

    startime=time.time()

    Mode_Name='SingleT'+str(modeType)+'FLS_'+modeName
    Mode=eval(Mode_Name+str((Rule_num,Antecedents_num,InitialSetup_List)))

    print('******************************************************************')
    #print(Mode_Name+'.variables',Mode.variables)
    print('******************************************************************')
    print(Mode_Name+'.trainable_variables:',Mode.trainable_variables)
    print('******************************************************************')

    if len(Xtrain)<batchSIZE or len(Xtrain)<processesNum:
        print('Warning! The number of training data must be greater than the number of batches and the number of possesser!')

    Block_SizeOfProcesses= len(Xtrain)//processesNum
 
    Loss_save = np.zeros(epoch)
    for epoch_id in range(epoch):
        print('>>>>>>>>>>>>>>>>>>>>>>>epoch:{}/{}<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(epoch_id+1,epoch))
        Epoch_sample_id=random.sample(range(0,len(Xtrain)),len(Xtrain))
        Xtrain_epoch = Xtrain[Epoch_sample_id,:]
        Ytrain_epoch = Ytrain[Epoch_sample_id]

        Grade_subMode = Queue()
        subModes = []
        for subMode_id in range(processesNum):
            submode = Process(target= SubMode_train ,args = (Mode,lossFunction,\
                Xtrain_epoch[subMode_id*Block_SizeOfProcesses:(subMode_id+1)*Block_SizeOfProcesses,:],\
                Ytrain_epoch[subMode_id*Block_SizeOfProcesses:(subMode_id+1)*Block_SizeOfProcesses], \
                batchSIZE,Grade_subMode,subMode_learningRate))
            subModes.append(submode)
            submode.start()

        for submode in subModes:
            submode.join()
        
        Grades_set = Grade_subMode.get()
        saveloss = Grades_set[1]
        for i_num in range(1,processesNum,1):
            q_g =  Grade_subMode.get()
            for j_num in range(len(q_g)):
                Grades_set[0][j_num] = Grades_set[0][j_num] + q_g[0][j_num]
                Grades_set[2][j_num] = Grades_set[2][j_num] + q_g[2][j_num]
            saveloss += q_g[1]
    
        for j_num in range(len(Grades_set)):
            Grades_set[0][j_num] = Grades_set[0][j_num] / processesNum
            Grades_set[2][j_num] = Grades_set[2][j_num] / processesNum
    
        Mode.Setting_parameters(Grades_set[0]) 
        optimizer.apply_gradients(zip(Grades_set[2],Mode.trainable_variables))

        Loss_save[epoch_id]= tf.sqrt(saveloss)

        print('epoch:{}/{},loss:{}'.format(epoch_id+1,epoch,saveloss))


        
    endtime=time.time()
    dtime=endtime-startime

    outputPredict=Mode(Xpredict)   
    Loss_predict=tf.sqrt(lossFunction(Ypredict,outputPredict))

    print('>>>>>>>>>>>>>>>>>>>>>>> The program has ended! Totial time:%.8f <<<<<<<<<<<<<<<<<<<<<<<<'%dtime)
   
    return Loss_save,Loss_predict,dtime


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

data_Num = 4
Rule = [16,32]
Epoch_num = 20
processes_num = [processes_N for processes_N in range(2,22,2)]
AntecedentsNum=4
data_size=500
predict_size = 300
TRAIN_XY=[]
TEST_XY=[]
for multiple in range(1,data_Num+1,1):
    X_train=np.zeros([multiple*data_size,AntecedentsNum])
    Y_train=np.zeros(multiple*data_size)
    X_test=np.zeros([predict_size,AntecedentsNum])
    Y_test=np.zeros(predict_size)

    X_train[:,0]=x_star[0:multiple*data_size]
    X_train[:,1]=x_star[1:multiple*data_size+1]
    X_train[:,2]=x_star[2:multiple*data_size+2]
    X_train[:,3]=x_star[3:multiple*data_size+3]
    Y_train=x_star[4:multiple*data_size+4]

    X_test[:,0]=x_star[multiple*data_size+1:multiple*data_size+predict_size+1]
    X_test[:,1]=x_star[multiple*data_size+2:multiple*data_size+predict_size+2]
    X_test[:,2]=x_star[multiple*data_size+3:multiple*data_size+predict_size+3]
    X_test[:,3]=x_star[multiple*data_size+4:multiple*data_size+predict_size+4]
    Y_test=x_star[multiple*data_size+5:multiple*data_size+predict_size+5]
    TRAIN_XY.append([X_train,Y_train])
    TEST_XY.append([X_test,Y_test])

# for i in range(4):
#     print('X_train{}.shape:{},Y_train{}.shape{}'.format(i+1,TRAIN_XY[i][0].shape,i+1,TRAIN_XY[i][1].shape))
#     print('X_test{}.shape:{},Y_test{}.shape{}'.format(i+1,TEST_XY[i][0].shape,i+1,TEST_XY[i][1].shape))
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

parallel_time = np.zeros([len(Rule),data_Num,len(processes_num)])
parallel_predict_RMSE = np.zeros([len(Rule),data_Num,len(processes_num)])
parallel_RMSE = np.zeros([len(Rule),data_Num,len(processes_num),Epoch_num])

for r in range(len(Rule)):
    for d in range(data_Num):
        for p in range(len(processes_num)):
            _RMSE, _predict_RMSE,_time = FLS_TrainFun_parallel_1(Rule[r],AntecedentsNum,LL,TRAIN_XY[d][0],TRAIN_XY[d][1],TEST_XY[d][0],TEST_XY[d][1],\
                modeName='Mamdani',modeType=2,predictMode=True,optimizer=tf.keras.optimizers.Adam(0.05),\
                lossFunction=tf.keras.losses.mean_squared_error,batchSIZE=16,epoch=Epoch_num,subMode_learningRate=tf.constant(0.01),processesNum=processes_num[p])
            parallel_time[r,d,p] = _time
            parallel_predict_RMSE[r,d,p] = _predict_RMSE
            parallel_RMSE[r,d,p,:] = _RMSE

np.save("parallel_time.npy",parallel_time)
np.save("parallel_predict_RMSE.npy",parallel_predict_RMSE)        
np.save("parallel_RMSE.npy",parallel_RMSE)