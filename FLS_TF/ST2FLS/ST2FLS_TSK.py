#/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/3/19
# @Author  : Wangjiawen

#from tensorflow.python.ops.control_flow_ops import case_v2
import sys
sys.path.append("..")
from MF.MF_T2 import *
import tensorflow as tf
import numpy as np


class ST2FLS_TSK(tf.keras.Model):
    #构造函数__init__
    def __init__(self,FuzzyRuleNum,FuzzyAntecedentsNum,InitialSetup_List):
        super(ST2FLS_TSK,self).__init__()
        self.Rule_num = FuzzyRuleNum
        self.Antecedents_num = FuzzyAntecedentsNum
        self.Init_SetupList = InitialSetup_List
        FuzzyRuleBase_weights,FRBparameterNum,C_init,S_init = self._initialize_weight(InitialSetup_List)
        self.FRB_weights = FuzzyRuleBase_weights
        self.FRB_parameterNum = FRBparameterNum 
        self.C = C_init
        self.S = S_init  

    #模糊规则库参数初始化
    def _initialize_weight(self,InitialSetup_List):
        FRB_ParaNum = tf.constant(0,tf.int32)
        FRB_ParaList = list()
        for i in range(self.Rule_num):
            for j in range(self.Antecedents_num):

                if InitialSetup_List[i][j] == 'G':
                    FRB_ParaList.append(3)
                    FRB_ParaNum +=3

        FRB_W = tf.Variable(tf.math.abs(tf.random.get_global_generator().normal(\
            shape=(FRB_ParaNum,))),trainable=True)
        C = tf.Variable(tf.abs(tf.random.get_global_generator().normal(shape=(\
            self.Rule_num,self.Antecedents_num+1))),trainable=True) #初始化c1
        S = tf.Variable(tf.abs(tf.random.get_global_generator().normal(shape=(\
            self.Rule_num,self.Antecedents_num+1))),trainable=True) #初始化c2  
        print('***********Initialization of fuzzy rule base parameters completed!*************')
        return FRB_W,FRB_ParaList,C,S

    def Setting_parameters(self,Grades_set):
        self.FRB_weights.assign(Grades_set[0])
        self.C.assign(Grades_set[1])
        self.S.assign(Grades_set[2])

    def GetFRB_weights(self):
        return self.FRB_weights,self.C,self.S

    def Compute_LeftPoint(self,c1,UU,LL):    
        c1_sort = tf.sort(c1,direction='ASCENDING')
        c1_index = tf.argsort(c1,direction='ASCENDING')
        UU_sort = tf.gather(UU,c1_index)
        LL_sort = tf.gather(LL,c1_index)
        l_out = 0
        s = 0
        s1 = 0
        b2=c1_sort
        s = tf.reduce_sum(tf.multiply(b2,LL_sort))
        s1 = tf.reduce_sum(LL_sort)
        l_out=s/s1
        for i in range(self.Rule_num):
            s += b2[i]*(UU_sort[i]-LL_sort[i])
            s1 += UU_sort[i]-LL_sort[i]
            l_out = tf.minimum(l_out,s/s1)

        return l_out

    def Compute_RightPoint(self,c2,UU,LL):
        c2_sort = tf.sort(c2,direction='ASCENDING')
        c2_index = tf.argsort(c2,direction='ASCENDING')
        UU_sort = tf.gather(UU,c2_index)
        LL_sort = tf.gather(LL,c2_index)
        r_out = 0
        s = 0
        s1 = 0
        b1=c2_sort
        s = tf.reduce_sum(tf.multiply(b1,UU_sort))
        s1 = tf.reduce_sum(UU_sort)
        r_out=s/s1
        for i in range(self.Rule_num):
            s += b1[i]*(LL_sort[i]-UU_sort[i])
            s1 += LL_sort[i]-UU_sort[i]
            r_out = tf.maximum(r_out,s/s1)

        return r_out

    def Post_output(self,x):
        c1=tf.reduce_sum(tf.multiply(x,self.C[:,1:]),1)+self.C[:,0]- \
            tf.reduce_sum(tf.multiply(tf.abs(x),self.S[:,1:]),1)-self.S[:,0]
        c2=tf.reduce_sum(tf.multiply(x,self.C[:,1:]),1)+self.C[:,0]+ \
            tf.reduce_sum(tf.multiply(tf.abs(x),self.S[:,1:]),1)+self.S[:,0]
        return c1,c2

    def call(self,input_data):
        #tf.keras.backend.set_floatx('float64')
        samples_num = input_data.shape[0]
        Output_Left=tf.ones(samples_num)
        Output_Right=tf.ones(samples_num)

        for sample_i in range(samples_num):
            input = input_data[sample_i] 
            #print('**//////** Number {},input(Sample):{}'.format(sample_i,input)) 
            UU=np.ones(self.Rule_num)
            LL=np.ones(self.Rule_num)
            for j in range(self.Rule_num):
                Uu = tf.constant(1.0)
                Ll = tf.constant(1.0) 
                for k in range(self.Antecedents_num):
                    locat_num = self.Antecedents_num*j+k
                    if self.Init_SetupList[j][k]=='G':
                        mu_small,mu_big = Gausstype2(input[k],self.FRB_weights[locat_num:locat_num+3])
                        #print('mu_small,mu_big',mu_small,mu_big)
                    Uu *= mu_big
                    Ll *= mu_small
                UU=tf.tensor_scatter_nd_update(UU,tf.constant([[j]]),[Uu])
                LL=tf.tensor_scatter_nd_update(LL,tf.constant([[j]]),[Ll])
                UU=tf.cast(UU,dtype=tf.float32)
                LL=tf.cast(LL,dtype=tf.float32)
            c1,c2=self.Post_output(input)

            Output_Left=tf.tensor_scatter_nd_update(Output_Left,tf.constant([[sample_i]]),\
                [self.Compute_LeftPoint(c1,UU,LL)])
            Output_Right=tf.tensor_scatter_nd_update(Output_Right,tf.constant([[sample_i]]),\
                [self.Compute_RightPoint(c2,UU,LL)])
        Output = (Output_Right+Output_Left)/2.0
        #print('Output:',Output)
        return Output


# 测试
# import numpy as np
# N=100
# train_data_x=np.random.random((N,6))
# train_data_y=np.random.random((N,1))

# LL=[['G','G','G','G','G','G'],['G','G','G','G','G','G'],['G','G','G','G','G','G'],['G','G','G','G','G','G']]

# FLS2=ST2FLS_TSK(4,6,LL)

# #print(FLS2.GetFRB_weights())
# print(FLS2.trainable_variables)
# print('*****************************')
# output=FLS2(train_data_x)
# print(output)











