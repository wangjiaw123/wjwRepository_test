#/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/4/1
# @Author  : Wangjiawen

import sys
sys.path.append("..")
from Membership_Function.MembershipFunction_T1 import *
import tensorflow as tf
import numpy as np

class SingleT1FLS_TSK(tf.keras.Model):
    #构造函数__init__
    def __init__(self,FuzzyRuleNum,FuzzyAntecedentsNum,InitialSetup_List):
        super(SingleT1FLS_TSK,self).__init__()
        self.Rule_num = FuzzyRuleNum
        self.Antecedents_num = FuzzyAntecedentsNum
        self.Init_SetupList = InitialSetup_List
        FuzzyRuleBase_weights,FRBparameterNum,C_init = self._initialize_weight(InitialSetup_List)
        self.FRB_weights = FuzzyRuleBase_weights
        self.FRB_parameterNum = FRBparameterNum 
        self.C = C_init

    def Setting_parameters(self,Grades_set):
        self.FRB_weights.assign(Grades_set[0])
        self.C.assign(Grades_set[1])

    #模糊规则库参数初始化
    def _initialize_weight(self,InitialSetup_List):
        FRB_ParaNum = tf.constant(0,tf.int32)
        FRB_ParaList = list()
        for i in range(self.Rule_num):
            for j in range(self.Antecedents_num):

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

        FRB_W = tf.Variable(tf.math.abs(tf.random.get_global_generator().normal(\
            shape=(FRB_ParaNum,))),trainable=True)
        C = tf.Variable(tf.abs(tf.random.get_global_generator().normal(shape=(\
            self.Rule_num,self.Antecedents_num+1))),trainable=True) #初始化c1

        print('***********Initialization of fuzzy rule base parameters completed!*************')
        return FRB_W,FRB_ParaList,C

    def Post_out(self,x):
        C_help = tf.reduce_sum(tf.multiply(x,self.C[:,1:]),1)+self.C[:,0]
        return C_help
       

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
                    else:         # InitialSetup_List[i][j] == 'Gauss1':
                        u_help = Gauss1mf(input[k],self.FRB_weights[locat_num:locat_num+2])                
                    uu*=u_help
                UU=tf.tensor_scatter_nd_update(UU,tf.constant([[i]]),[uu]) 
            C_help = self.Post_out(input)   
            Output = tf.tensor_scatter_nd_update(Output,tf.constant([[sample_i]]),
                [tf.reduce_sum(tf.multiply(UU,C_help)/tf.reduce_sum(UU))])
        return Output




# # 测试
# import numpy as np
# N=100
# train_data_x=np.random.random((N,6))
# train_data_y=np.random.random((N,1))

# LL=[['G','G','G','G','G','G'],['G','G','G','G','G','G'],['G','G','G','G','G','G'],['G','G','G','G','G','G']]

# FLS1_tsk=SingleT1FLS_TSK(4,6,LL)

# #print(FLS1_tsk.GetFRB_weights())
# print(FLS1_tsk.trainable_variables)
# print('*****************************')
# output=FLS1_tsk(train_data_x)
# print(output)


