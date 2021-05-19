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


class ST2FLS_FWA(tf.keras.Model):
    #构造函数__init__
    def __init__(self,FuzzyRuleNum,FuzzyAntecedentsNum,InitialSetup_List):
        super(ST2FLS_FWA,self).__init__()
        self.Rule_num = FuzzyRuleNum
        self.Antecedents_num = FuzzyAntecedentsNum
        self.Init_SetupList = InitialSetup_List
        FuzzyRuleBase_weights,FRBparameterNum,W_m_init,W_s_init,B_m_init,B_s_init = \
            self._initialize_weight(InitialSetup_List)
        self.FRB_weights = FuzzyRuleBase_weights
        self.FRB_parameterNum = FRBparameterNum 
        self.W_m = W_m_init
        self.W_s = W_s_init  
        self.B_m = B_m_init
        self.B_s = B_s_init

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
        W_m_init = tf.Variable(tf.abs(tf.random.get_global_generator().normal(shape=(\
            self.Rule_num,self.Antecedents_num))),trainable=True) #初始化
        W_s_init = tf.Variable(tf.abs(tf.random.get_global_generator().normal(shape=(\
            self.Rule_num,self.Antecedents_num))),trainable=True) #初始化 
        B_m_init = tf.Variable(tf.abs(tf.random.get_global_generator().normal(shape=(\
            self.Rule_num,))),trainable=True)
        B_s_init = tf.Variable(tf.abs(tf.random.get_global_generator().normal(shape=(\
            self.Rule_num,))),trainable=True)             
        print('***********Initialization of fuzzy rule base parameters completed!*************')
        return FRB_W,FRB_ParaList,W_m_init,W_s_init,B_m_init,B_s_init

    def Setting_parameters(self,Grades_set):
        self.FRB_weights.assign(Grades_set[0])
        self.W_m.assign(Grades_set[1])
        self.W_s.assign(Grades_set[2])
        self.B_m.assign(Grades_set[3])
        self.B_s.assign(Grades_set[4])

    def GetFRB_weights(self):
        return self.FRB_weights,self.W_m,self.W_s,self.B_m,self.B_s

    def Compute_LeftPoint(self,c1,UU,LL,n):   
        #print('<<<<<<<<<<<<')
        c1_sort = tf.sort(c1,direction='ASCENDING')
        c1_index = tf.argsort(c1,direction='ASCENDING')
        #print('c1_index,UU,LL',c1_index,UU,LL)
        UU_sort = tf.gather(UU,c1_index)
        #print('*****************',UU,c1_index)
        LL_sort = tf.gather(LL,c1_index)
        l_out = 0
        s = 0
        s1 = 0
        b2=c1_sort
        #print('<<<<<<<<<<<<')
        s = tf.reduce_sum(tf.multiply(b2,LL_sort))
        s1 = tf.reduce_sum(LL_sort)+0.0000001
        l_out=s/s1
        for i in range(n):
            s += b2[i]*(UU_sort[i]-LL_sort[i])
            s1 += UU_sort[i]-LL_sort[i]
            l_out = tf.minimum(l_out,s/s1)

        return l_out

    def Compute_RightPoint(self,c2,UU,LL,n):
        c2_sort = tf.sort(c2,direction='ASCENDING')
        c2_index = tf.argsort(c2,direction='ASCENDING')
        UU_sort = tf.gather(UU,c2_index)
        LL_sort = tf.gather(LL,c2_index)
        r_out = 0
        s = 0
        s1 = 0
        b1=c2_sort
        s = tf.reduce_sum(tf.multiply(b1,UU_sort))
        s1 = tf.reduce_sum(UU_sort)+0.0000001
        r_out=s/s1
        for i in range(n):
            s += b1[i]*(LL_sort[i]-UU_sort[i])
            s1 += LL_sort[i]-UU_sort[i]
            r_out = tf.maximum(r_out,s/s1)

        return r_out

    def Post_output_y(self,UU,LL,k):
        y_small = tf.constant(0)
        y_big = tf.constant(0)    
        #print('>>>****************************************************************')
        #print('k',k,LL,self.W_m[k,:]+self.W_s[k,:],self.W_m[k,:]-self.W_s[k,:])
        y_small=self.Compute_LeftPoint(LL,self.W_m[k,:]+self.W_s[k,:],\
                    self.W_m[k,:]-self.W_s[k,:],self.Antecedents_num)+self.B_m[k]+self.B_s[k]
        #print('y_small',y_small)
        y_big=self.Compute_RightPoint(UU,self.W_m[k,:]+self.W_s[k,:],\
                    self.W_m[k,:]-self.W_s[k,:],self.Antecedents_num)+self.B_m[k]-self.B_s[k] 
        #print('>>****************************************************************')
        return y_small,y_big                  
 
    def call(self,input_data):
        #tf.keras.backend.set_floatx('float64')
        samples_num = input_data.shape[0]
        #Output_Left=tf.Variable(tf.zeros((samples_num,)))
        #Output_Right=tf.Variable(tf.zeros((samples_num,)))
        Output_Left=tf.ones(samples_num)
        Output_Right=tf.ones(samples_num)

        for sample_i in range(samples_num):
            input = input_data[sample_i] 
            #print('**//////** Number {},input(Sample):{}'.format(sample_i,input)) 
            UU=tf.ones(self.Rule_num)
            LL=tf.ones(self.Rule_num)
            y_small=tf.ones(self.Rule_num)
            y_big=tf.ones(self.Rule_num)

            for j in range(self.Rule_num):
                Uu = tf.constant(1.0)
                Ll = tf.constant(1.0) 
                Uuu=tf.ones(self.Antecedents_num)
                Lll=tf.ones(self.Antecedents_num)
                for k in range(self.Antecedents_num):
                    locat_num = self.Antecedents_num*j+k
                    #隶属函数还可以再添加
                    if self.Init_SetupList[j][k]=='G':
                        mu_small,mu_big = Gausstype2(input[k],self.FRB_weights[locat_num:locat_num+3])
                        #print('mu_small,mu_big',mu_small,mu_big)
                    Uu *= mu_big
                    Ll *= mu_small
                    Uuu=tf.tensor_scatter_nd_update(Uuu,tf.constant([[k]]),[mu_big])
                    Lll=tf.tensor_scatter_nd_update(Lll,tf.constant([[k]]),[mu_small])
                #print('Uu,Ll:',Uu,Ll)
                #print('Uu.shape:',tf.shape(Uu))
                y_small0,y_big0=self.Post_output_y(Uuu,Lll,j)
                #print('y_small0,y_big0',y_small0,y_big0)
                y_small=tf.tensor_scatter_nd_update(y_small,tf.constant([[j]]),[y_small0])
                y_big=tf.tensor_scatter_nd_update(y_big,tf.constant([[j]]),[y_big0])
                #print('+++++++++',y_big,y_small)

                UU=tf.tensor_scatter_nd_update(UU,tf.constant([[j]]),[Uu])
                LL=tf.tensor_scatter_nd_update(LL,tf.constant([[j]]),[Ll])
                UU=tf.cast(UU,dtype=tf.float32)
                LL=tf.cast(LL,dtype=tf.float32)
            #print('+++++//////////+++++++UU,LL:',UU,LL)
            #################
            
            #print('y_small,y_big',y_small,y_big)

            Output_Left=tf.tensor_scatter_nd_update(Output_Left,tf.constant([[sample_i]]),\
                [self.Compute_LeftPoint(y_small,UU,LL,self.Rule_num)])
            Output_Right=tf.tensor_scatter_nd_update(Output_Right,tf.constant([[sample_i]]),\
                [self.Compute_RightPoint(y_big,UU,LL,self.Rule_num)])

            #Output_Left[sample_i]=self.Compute_LeftPoint(UU,LL)
            #Output_Right[sample_i]=self.Compute_RightPoint(UU,LL)
        #Output = tf.divide(tf.add(Output_Left,Output_Right),2.0)
        Output = (Output_Right+Output_Left)/2.0
        #print('Output:',Output)
        return Output


# #测试
# import numpy as np
# N=100
# train_data_x=np.random.random((N,6))
# train_data_y=np.random.random((N,1))

# LL=[['G','G','G','G','G','G'],['G','G','G','G','G','G'],['G','G','G','G','G','G'],['G','G','G','G','G','G']]

# FLS2=ST2FLS_FWA(4,6,LL)

# #print(FLS2.GetFRB_weights())
# print(FLS2.trainable_variables)
# print('*****************************')
# output=FLS2(train_data_x)
# print(output)




