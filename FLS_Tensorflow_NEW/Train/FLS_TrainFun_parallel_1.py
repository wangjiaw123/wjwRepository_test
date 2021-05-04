#/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/5/4
# @Author  : Wangjiawen

# Note : Parallel algorithm must be run on Linux system.



import sys

#from tensorflow.python.framework.load_library import load_file_system_library
sys.path.append('..')
import os
import time
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
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
    modeName='Mamdani',modeType=2,predictMode=True,validationRatio=0.1,XvalidationSet=None,YvalidationSet=None,\
    optimizer=tf.keras.optimizers.Adam(0.05),lossFunction=tf.keras.losses.mean_squared_error,\
    batchSIZE=1,epoch=5,subMode_learningRate=tf.constant(0.01),useGPU=False,processesNum=None,RMSE_threshold=None):

    '''
    Rule_num:规则数量,Antecedents_num:前件数量,InitialSetup_List:模糊规则初始化列表
    Xtrain,Ytrain,表示训练数据的输入和相应的标签
    batchSIZE:批量大小,useGPU:设置是否使用GPU训练模型,saveMode:设置是否保存模型,
    modeName:模型的命名(后缀名为.h5,例如'mode.h5'),modeSavePath:设置保存模型的路径.
    '''
    startime=time.time()
    if useGPU:
        gpus = tf.config.list_physical_devices("GPU")
        print(gpus)
        if gpus:
            gpu0 = gpus[0] #如果有多个GPU，仅使用第0个GPU
            tf.config.experimental.set_memory_growth(gpu0, True) #设置GPU显存用量按需使用
            # 或者也可以设置GPU显存为固定使用量(例如：4G)
            #tf.config.experimental.set_virtual_device_configuration(gpu0,
            #    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]) 
            tf.config.set_visible_devices([gpu0],"GPU")

    Mode_Name='SingleT'+str(modeType)+'FLS_'+modeName
    Mode=eval(Mode_Name+str((Rule_num,Antecedents_num,InitialSetup_List)))

    print('******************************************************************')
    #print(Mode_Name+'.variables',Mode.variables)
    print('******************************************************************')
    print(Mode_Name+'.trainable_variables:',Mode.trainable_variables)
    print('******************************************************************')

    if len(Xtrain)<batchSIZE or len(Xtrain)<processesNum:
        print('Warning! The number of training data must be greater than the number of batches and the number of possesser!')

    # if (len(Xtrain))%processesNum != 0:
    #     LACK_dataNum = processesNum - (len(Xtrain))%processesNum
    #     supplement_sample_id = random.sample(range(0,len(Xtrain)),LACK_dataNum)
    #     Xtrain = np.r_[Xtrain,Xtrain[supplement_sample_id,:]]
    #     Ytrain = np.r_[Ytrain,Ytrain[supplement_sample_id]]

    Block_SizeOfProcesses= len(Xtrain)//processesNum
    validation_sample_id=random.sample(range(0,len(Xtrain)),int(len(Xtrain)*validationRatio))
    
    if (XvalidationSet == None) and (YvalidationSet == None):
        XvalidationSet=Xtrain[validation_sample_id,:]
        YvalidationSet=Ytrain[validation_sample_id]

    Loss_save = np.zeros(epoch)
    Loss_validat = np.zeros(epoch)
    RMSE_threshold_count = 0
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
                #tf.assign_add(Grades_set[0][j_num] , q_g[0][j_num])
            saveloss += q_g[1]
        #Grades_H = Grades_set[0]
        for j_num in range(len(Grades_set)):
            Grades_set[0][j_num] = Grades_set[0][j_num] / processesNum
            Grades_set[2][j_num] = Grades_set[2][j_num] / processesNum
    
        Mode.Setting_parameters(Grades_set[0]) 
        optimizer.apply_gradients(zip(Grades_set[2],Mode.trainable_variables))
                   
        # optimizer.apply_gradients(zip(Grades_set[0],Mode.trainable_variables))

        #print('Grades_set',Grades_set)

        #saveloss+=tf.reduce_sum(Loss).numpy()

        #print('>>>>>>>>>> epoch:{}/{},Block_SizeOfProcesses:{},block_loss:{}'.format(epoch_id+1,epoch,Block_id+1,Block_SizeOfProcesses,Loss))
            #print('**********grades:',grades)
        Loss_save[epoch_id]= tf.sqrt(saveloss)

        output_data_validat=Mode(XvalidationSet)
        Loss_validat[epoch_id]=tf.sqrt(lossFunction(output_data_validat,YvalidationSet))
        print('epoch:{}/{},loss:{},loss_validat:{}'.format(epoch_id+1,epoch,saveloss,Loss_validat[epoch_id]))

        # 3次小于阈值则结束
        if RMSE_threshold and Loss_save[epoch_id] < RMSE_threshold:
            RMSE_threshold_count += 1
        if RMSE_threshold and RMSE_threshold_count>=3:
            epoch = epoch_id
            break
            

    endtime=time.time()
    dtime=endtime-startime
    print('>>>>>>>>>>>>>>>>>>>>>>> Totial time:%.8f <<<<<<<<<<<<<<<<<<<<<<<<'%dtime)
    print('Please wait a moment,calculating output ...... ')

    time_now = time.strftime("%Y%m%d-%H%M",time.localtime())
    #SavePath = os.path.split(__file__)[0]+'/'+time_now
    SavePath = os.getcwd()+'/'+time_now
    
    if not os.path.exists(SavePath):
        os.makedirs(SavePath)
        print('*************File save path:{}*************'.format(SavePath))
    Mode.save_weights(SavePath+"/"+Mode_Name+".ckpt")    #保存模型
    #使用  Mode.load_weights('/*/*/*.ckpt')   从参数文件中读取数据并写入当前网络，即恢复之前训练的FLS模型 


    plt.figure()
    plt.plot(np.linspace(1,epoch,epoch,dtype=int),Loss_save)
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('RMSE(Train)')
    #plt.title('RMSE(Train)')
    plt.savefig(SavePath+"/"+"RMSE(Train).pdf", format="pdf", dpi=300, bbox_inches="tight")

    plt.figure()
    plt.plot(np.linspace(1,epoch,epoch,dtype=int),Loss_validat)
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('Validat RMSE(Train)')
    plt.savefig(SavePath+"/"+"Validat RMSE(Train).pdf", format="pdf", dpi=300, bbox_inches="tight")


##############################
    plt.figure()
    plt.plot(np.linspace(1,len(Xtrain),len(Xtrain)),Ytrain)
    plt.grid(True)
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('Real(train)')    
    plt.savefig(SavePath+"/"+"Real(train).pdf", format="pdf", dpi=300, bbox_inches="tight")



    outputTrain=Mode(Xtrain)
    plt.figure()
    plt.plot(np.linspace(1,len(Xtrain),len(Xtrain)),Ytrain)
    plt.plot(np.linspace(1,len(Xtrain),len(Xtrain)),outputTrain)
    plt.grid(True)
    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend(["Real","Predict"], loc=1)
    plt.title('Real and predict(train)')    
    plt.savefig(SavePath+"/"+"Real and predict(train).pdf", format="pdf", dpi=300, bbox_inches="tight")

    if predictMode:
        outputPredict=Mode(Xpredict)
        return outputPredict
    else:
        outputPredict=Mode(Xpredict)   
        Loss_predict=lossFunction(Ypredict,outputPredict)
        print('Predict Mode,the predict loss:{}'.format(Loss_predict))
        plt.figure()
        plt.grid(True)
        plt.plot(np.linspace(len(Xtrain)+1,len(Xpredict)+len(Xtrain)+1,len(Xpredict)),Ypredict)
        plt.plot(np.linspace(len(Xtrain)+1,len(Xpredict)+len(Xtrain)+1,len(Xpredict)),outputPredict)
        plt.xlabel('t')
        plt.ylabel('y')
        plt.legend(["Real","Predict"], loc=1)
        plt.title('Real and predict(predict)') 
        plt.savefig(SavePath+"/"+"Real and predict(predict).pdf", format="pdf", dpi=300, bbox_inches="tight")
    
    
    #plt.show()

    print('********************* The program has ended! ***********************')
    
