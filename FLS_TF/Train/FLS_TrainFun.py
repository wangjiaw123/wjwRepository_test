#/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/3/20
# @Author  : Wangjiawen

import sys
sys.path.append('..')

import time
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.python.eager.backprop import GradientTape
from tensorflow.python.ops.functional_ops import Gradient
#导入2型FLS系统
from ST2FLS.ST2FLS_Mamdani import *
from ST2FLS.ST2FLS_TSK import *
from ST2FLS.ST2FLS_FWA import *
#导入1型FLS系统
from ST1FLS.ST1FLS_Mamdani import *
from ST1FLS.ST1FLS_TSK import *

def FLS_TrainFun(Rule_num,Antecedents_num,InitialSetup_List,Xtrain,Ytrain,Xpredict,Ypredict=None,\
    modeName='Mamdani',modeType=2,predictMode=True,validationRatio=0.1,XvalidationSet=None,YvalidationSet=None,\
    optimizer=tf.keras.optimizers.Adam(0.05),lossFunction=tf.keras.losses.mean_squared_error,\
    batchSIZE=1,epoch=5,useGPU=False,saveMode=False,outputModeName=None,modeSavePath=None):

    '''
    Rule_num:规则数量,Antecedents_num:前件数量,InitialSetup_List:模糊规则初始化列表，Xtrain,Ytrain,表示训练数据的输入和相应的标签，
    Xpredict表示测试数据，Ypredict表示测试数据的标签(与predictMode参数同时起作用，若predictMode=True，则Xpredict,Ypredict=None，
    若predictMode=False，则Xpredict,Ypredict需要传入数据)，batchSIZE:批量大小,epoch设置训练轮数,useGPU:设置是否使用GPU训练模型,
    modeName:模型的命名,如果modeType=1,modeName处可以选择'Mamdani'、'TSK';如果modeType=2,modeName处可以选择'Mamdani'、'TSK'、'FWA'.
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

    Mode_Name='ST'+str(modeType)+'FLS_'+modeName
    Mode=eval(Mode_Name+str((Rule_num,Antecedents_num,InitialSetup_List)))

    print('******************************************************************')
    #print('FLS2.variables',Mode.variables)
    print('******************************************************************')
    print('FLS2.trainable_variables:',Mode.trainable_variables)
    print('******************************************************************')

    if len(Xtrain)<batchSIZE:
        print('Warning! The number of training data must be greater than the number of batches!')
    Block_size=len(Xtrain)//batchSIZE

    validation_sample_id=random.sample(range(0,(len(Xtrain)-len(Xtrain)%batchSIZE)),int(len(Xtrain)*validationRatio))
    
    if (XvalidationSet == None) and (YvalidationSet == None):
        XvalidationSet=Xtrain[validation_sample_id,:]
        YvalidationSet=Ytrain[validation_sample_id]

    Loss_save = np.zeros(epoch)
    Loss_validat = np.zeros(epoch)
    for epoch_id in range(epoch):
        saveloss=0.0
        for Block_id in range(Block_size):
            with tf.GradientTape() as tape:
                #print('--------------',Xtrain[Block_id*batchSIZE:Block_id*batchSIZE+batchSIZE,:])
                output_data=Mode(Xtrain[Block_id*batchSIZE:Block_id*batchSIZE+batchSIZE,:])
                #print('output_data',output_data)
                Loss=lossFunction(output_data,Ytrain[Block_id*batchSIZE:Block_id*batchSIZE+batchSIZE])
                Loss=tf.clip_by_value(Loss,0.0000001,10)
            grades=tape.gradient(Loss,Mode.trainable_variables)
            optimizer.apply_gradients(zip(grades,Mode.trainable_variables))
            saveloss+=tf.reduce_sum(Loss).numpy()
            print('>>>>>>>>>> epoch:{}/{},block_id:{},block_size:{},block_loss:{}'.format(epoch_id+1,epoch,Block_id+1,Block_size,Loss))
            #print('**********grades:',grades)
        Loss_save[epoch_id]=tf.sqrt(saveloss/len(Xtrain))

        output_data_validat=Mode(XvalidationSet)
        Loss_validat[epoch_id]=tf.sqrt(lossFunction(output_data_validat,YvalidationSet)/len(XvalidationSet))
        print('epoch:{}/{},Loss:{},Loss_validat:{}'.format(epoch_id+1,epoch,Loss,Loss_validat[epoch_id]))
    
    endtime=time.time()
    dtime=endtime-startime
    print('>>>>>>>>>>>>>>>>>>>>>>> Totial time:%.8f <<<<<<<<<<<<<<<<<<<<<<<<'%dtime)
    print('Please wait a moment,calculating output ...... ')

    if saveMode:
        Mode.save(filepath=modeSavePath+'\\'+outputModeName)
    plt.figure()
    plt.plot(np.linspace(1,epoch,epoch,dtype=int),Loss_save)
    plt.xlabel('epoch')
    plt.ylabel('RMSE(Train)')
    #plt.title('RMSE(Train)')

    plt.figure()
    plt.plot(np.linspace(1,epoch,epoch,dtype=int),Loss_validat)
    plt.xlabel('epoch')
    plt.ylabel('Validat RMSE(Train)')

    outputTrain=Mode(Xtrain)
    plt.figure()
    plt.plot(np.linspace(1,len(Xtrain),len(Xtrain)),Ytrain)
    plt.plot(np.linspace(1,len(Xtrain),len(Xtrain)),outputTrain)
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('Real and predict(train)')    

    if predictMode:
        outputPredict=Mode(Xpredict)
        return outputPredict
    else:
        outputPredict=Mode(Xpredict)   
        Loss_predict=tf.sqrt(lossFunction(Ypredict,outputPredict)/len(Xpredict))
        print('Predict Mode,the predict loss:{}'.format(Loss_predict))
        plt.figure()
        plt.plot(np.linspace(len(Xtrain)+1,len(Xpredict)+len(Xtrain)+1,len(Xpredict)),Ypredict)
        plt.plot(np.linspace(len(Xtrain)+1,len(Xpredict)+len(Xtrain)+1,len(Xpredict)),outputPredict)
        plt.xlabel('t')
        plt.ylabel('y')
        plt.title('Real and predict(predict)') 
    
    
    plt.show()

    print('********************* Training and predicting are all over! ***********************')








