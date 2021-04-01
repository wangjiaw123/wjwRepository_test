#/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/4/1
# @Author  : Wangjiawen

import sys
sys.path.append("..")
from Membership_Function.MembershipFunction_T1 import *
import tensorflow as tf
#import numpy as np
#import pp

class SingleT1FLS_Mamdani(tf.keras.Model):
    #构造函数__init__
    def __init__(self,FuzzyRuleNum,FuzzyAntecedentsNum,InitialSetup_List):
        super(SingleT1FLS_Mamdani,self).__init__()
        self.Rule_num = FuzzyRuleNum
        self.Antecedents_num = FuzzyAntecedentsNum
        self.Init_SetupList = InitialSetup_List
        FuzzyRuleBase_weights,FRBparameterNum,c1_init = self._initialize_weight(InitialSetup_List)
        self.FRB_weights = FuzzyRuleBase_weights
        self.FRB_parameterNum = FRBparameterNum 
        self.c1 = c1_init

    #模糊规则库参数初始化
    def _initialize_weight(self,InitialSetup_List):
        FRB_ParaNum = tf.constant(0,tf.int32)
        FRB_ParaList = list()
        for i in range(self.Rule_num):
            for j in range(self.Antecedents_num):
                #默认是高斯1型隶属函数
                if self.Init_SetupList[i][j] == 'Gauss2':
                    FRB_ParaList.append(4)
                    FRB_ParaNum +=4
                elif self.Init_SetupList[i][j] == 'Trap':
                    FRB_ParaList.append(4)
                    FRB_ParaNum +=4
                elif self.Init_SetupList[i][j] == 'Tri':
                    FRB_ParaList.append(3)
                    FRB_ParaNum +=3
                elif self.Init_SetupList[i][j] == 'Sig':
                    FRB_ParaList.append(2)
                    FRB_ParaNum +=2
                elif self.Init_SetupList[i][j] == 'Gbell':
                    FRB_ParaList.append(3)
                    FRB_ParaNum +=3
                elif self.Init_SetupList[i][j] == 'Psig':
                    FRB_ParaList.append(4)
                    FRB_ParaNum +=4
                elif self.Init_SetupList[i][j] == 'Dsig':
                    FRB_ParaList.append(4)
                    FRB_ParaNum +=4
                else:         # self.Init_SetupList[i][j] == 'Gauss1':
                    FRB_ParaList.append(2)
                    FRB_ParaNum +=2

        FRB_W = tf.Variable(tf.math.abs(tf.random.get_global_generator().normal(shape=(FRB_ParaNum,))),trainable=True)
        c1 = tf.Variable(tf.abs(tf.random.get_global_generator().normal(shape=(self.Rule_num,))),trainable=True)     #初始化c1

        print('***********Initialization of fuzzy rule base parameters completed!*************')
        return FRB_W,FRB_ParaList,c1  
    '''
    def compute_output(self,input_data,Output):
        samples_num = input_data.shape[0]
        #Output=tf.ones(samples_num)

        for sample_i in range(samples_num):
            input = input_data[sample_i] 
            UU=tf.ones(self.Rule_num)
            for i in range(self.Rule_num):
                uu=tf.constant(1.0)
                for k in range(self.Antecedents_num):
                    locat_num = self.Antecedents_num*i+k
                    #默认是高斯1型隶属函数
                    if self.Init_SetupList[i][k] == 'Gauss2':
                        u_help = Gauss2mf(input[k],self.FRB_weights[locat_num:locat_num+4])
                    elif self.Init_SetupList[i][k] == 'Trap':
                        u_help = Trapmf(input[k],self.FRB_weights[locat_num:locat_num+4])
                    elif self.Init_SetupList[i][k] == 'Tri':
                        u_help = Trimf(input[k],self.FRB_weights[locat_num:locat_num+3])
                    elif self.Init_SetupList[i][k] == 'Sig':
                        u_help = Sigmf(input[k],self.FRB_weights[locat_num:locat_num+2])
                    elif self.Init_SetupList[i][k] == 'Gbell':
                        u_help = Gbellmf(input[k],self.FRB_weights[locat_num:locat_num+3])
                    elif self.Init_SetupList[i][k] == 'Psig':
                        u_help = Psigmf(input[k],self.FRB_weights[locat_num:locat_num+4])
                    elif self.Init_SetupList[i][k] == 'Dsig':
                        u_help = Dsigmf(input[k],self.FRB_weights[locat_num:locat_num+4])
                    else:         # self.Init_SetupList[i][j] == 'Gauss1':
                        u_help = Gauss1mf(input[k],self.FRB_weights[locat_num:locat_num+2])                
                    uu*=u_help
                UU=tf.tensor_scatter_nd_update(UU,tf.constant([[i]]),[uu])    
            Output = tf.tensor_scatter_nd_update(Output,tf.constant([[sample_i]]), 
                [tf.reduce_sum(tf.multiply(UU,self.c1)/tf.reduce_sum(UU))])
    '''
    def call(self,input_data):
        samples_num = input_data.shape[0]
        Output=tf.ones(samples_num)

        for sample_i in range(samples_num):
            input = input_data[sample_i] 
            UU=tf.ones(self.Rule_num)
            for i in range(self.Rule_num):
                uu=tf.constant(1.0)
                for k in range(self.Antecedents_num):
                    locat_num = self.Antecedents_num*i+k
                    #默认是高斯1型隶属函数
                    if self.Init_SetupList[i][k] == 'Gauss2':
                        u_help = Gauss2mf(input[k],self.FRB_weights[locat_num:locat_num+4])
                    elif self.Init_SetupList[i][k] == 'Trap':
                        u_help = Trapmf(input[k],self.FRB_weights[locat_num:locat_num+4])
                    elif self.Init_SetupList[i][k] == 'Tri':
                        u_help = Trimf(input[k],self.FRB_weights[locat_num:locat_num+3])
                    elif self.Init_SetupList[i][k] == 'Sig':
                        u_help = Sigmf(input[k],self.FRB_weights[locat_num:locat_num+2])
                    elif self.Init_SetupList[i][k] == 'Gbell':
                        u_help = Gbellmf(input[k],self.FRB_weights[locat_num:locat_num+3])
                    elif self.Init_SetupList[i][k] == 'Psig':
                        u_help = Psigmf(input[k],self.FRB_weights[locat_num:locat_num+4])
                    elif self.Init_SetupList[i][k] == 'Dsig':
                        u_help = Dsigmf(input[k],self.FRB_weights[locat_num:locat_num+4])
                    else:         # self.Init_SetupList[i][j] == 'Gauss1':
                        u_help = Gauss1mf(input[k],self.FRB_weights[locat_num:locat_num+2])                
                    uu*=u_help
                UU=tf.tensor_scatter_nd_update(UU,tf.constant([[i]]),[uu])    
            Output = tf.tensor_scatter_nd_update(Output,tf.constant([[sample_i]]), 
                [tf.reduce_sum(tf.multiply(UU,self.c1)/tf.reduce_sum(UU))])       
        return Output
     
    '''
    def call(self,input_data):
        ncpus = 6
        samples_num = input_data.shape[0]
        Output=tf.ones(samples_num)
        data_num=(samples_num-samples_num%ncpus)/ncpus
        inputs=[]
        for i in range(ncpus):
            inputs.append(input_data[data_num*i:(i+1)*data_num])
    
        # Creates jobserver with ncpus workers
        job_server = pp.Server(ncpus, ppservers=ppservers)
        print("Starting pp with", job_server.get_ncpus(), "workers")

        jobs = [(input, job_server.submit(self.compute_output,(input,Output), (compute_output,), ("tensorflow",))) for input in inputs]
        for _,job in jobs:
            job()
        return Output
    '''



# # 测试
# import numpy as np
# N=100
# train_data_x=np.random.random((N,6))
# train_data_y=np.random.random((N,1))

# LL=[['G','G','G','G','G','G'],['G','G','G','G','G','G'],['G','G','G','G','G','G'],['G','G','G','G','G','G']]

# FLS1_Mamdani=SingleT1FLS_Mamdani(4,6,LL)

# print(FLS1_Mamdani.GetFRB_weights())
# print(FLS1_Mamdani.trainable_variables)
# print('*****************************************************************')
# output=FLS1_Mamdani(train_data_x)
# print(output)





