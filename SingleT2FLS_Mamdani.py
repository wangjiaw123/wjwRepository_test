#/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/3/19
# @Author  : Wangjiawen


from MembershipFunction import *
import tensorflow as tf
import numpy as np


class SingleT2FLS_Mamdani(tf.keras.Model):
    #构造函数__init__
    def __init__(self,FuzzyRuleNum,FuzzyAntecedentsNum,InitialSetup_List):
        super(SingleT2FLS_Mamdani,self).__init__()
        self.Rule_num = FuzzyRuleNum
        self.Antecedents_num = FuzzyAntecedentsNum
        self.Init_SetupList = InitialSetup_List
        FuzzyRuleBase_weights,FRBparameterNum,c1_init,c2_init = self._initialize_weight(InitialSetup_List)
        self.FRB_weights = FuzzyRuleBase_weights
        self.FRB_parameterNum = FRBparameterNum 
        self.c1 = c1_init
        self.c2 = c2_init 
        self.count = 0    

    #模糊规则库参数初始化
    def _initialize_weight(self,InitialSetup_List):
        FRB_ParaNum = tf.constant(0,tf.int32)
        FRB_ParaList = list()
        for i in range(self.Rule_num):
            for j in range(self.Antecedents_num):
                if InitialSetup_List[i][j] == 'G':
                    FRB_ParaList.append(3)
                    FRB_ParaNum +=3
        FRB_W = tf.Variable(tf.math.abs(tf.random.get_global_generator().normal(shape=(FRB_ParaNum,))),trainable=True)
        c1 = tf.Variable(tf.abs(tf.random.get_global_generator().normal(shape=(self.Rule_num,))),trainable=True)     #初始化c1
        c2 = tf.Variable(tf.abs(tf.random.get_global_generator().normal(shape=(self.Rule_num,))) \
                  ,trainable=True)  #初始化c2   #+c1[tf.argmax(c1,0)]-c1[tf.argmin(c1,0)] 
        print('***********Initialization of fuzzy rule base parameters completed!*************')
        return FRB_W,FRB_ParaList,c1,c2

  

    def GetFRB_weights(self):
        return self.FRB_weights,self.c1,self.c2

    def Compute_LeftPoint(self,UU,LL):    
        c1_sort = tf.sort(self.c1,direction='ASCENDING')
        c1_index = tf.argsort(self.c1,direction='ASCENDING')
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

    def Compute_RightPoint(self,UU,LL):
        c2_sort = tf.sort(self.c2,direction='ASCENDING')
        c2_index = tf.argsort(self.c2,direction='ASCENDING')
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
            UU=np.ones(self.Rule_num)
            LL=np.ones(self.Rule_num)
            for j in range(self.Rule_num):
                Uu = tf.constant(1.0)
                Ll = tf.constant(1.0) 
                for k in range(self.Antecedents_num):
                    locat_num = self.Antecedents_num*j+k
                    #隶属函数还可以再添加
                    if self.Init_SetupList[j][k]=='G':
                        mu_small,mu_big = Gausstype2(input[k],self.FRB_weights[locat_num:locat_num+3])
                        #print('mu_small,mu_big',mu_small,mu_big)
                    Uu *= mu_big
                    Ll *= mu_small

                #print('Uu,Ll:',Uu,Ll)
                #print('Uu.shape:',tf.shape(Uu))
                UU=tf.tensor_scatter_nd_update(UU,tf.constant([[j]]),[Uu])
                LL=tf.tensor_scatter_nd_update(LL,tf.constant([[j]]),[Ll])
                UU=tf.cast(UU,dtype=tf.float32)
                LL=tf.cast(LL,dtype=tf.float32)
            #print('+++++//////////+++++++UU,LL:',UU,LL)

            Output_Left=tf.tensor_scatter_nd_update(Output_Left,tf.constant([[sample_i]]),\
                [self.Compute_LeftPoint(UU,LL)])
            Output_Right=tf.tensor_scatter_nd_update(Output_Right,tf.constant([[sample_i]]),\
                [self.Compute_RightPoint(UU,LL)])

            #Output_Left[sample_i]=self.Compute_LeftPoint(UU,LL)
            #Output_Right[sample_i]=self.Compute_RightPoint(UU,LL)
        #Output = tf.divide(tf.add(Output_Left,Output_Right),2.0)
        Output = (Output_Right+Output_Left)/2.0
        #print('Output:',Output)
        return Output


# 测试
# import numpy as np
# N=100
# train_data_x=np.random.random((N,6))
# train_data_y=np.random.random((N,1))

# LL=[['G','G','G','G','G','G'],['G','G','G','G','G','G'],['G','G','G','G','G','G'],['G','G','G','G','G','G']]

# FLS2=SingleT2FLS_Mamdani(4,6,LL)

# #print(FLS2.GetFRB_weights())
# print(FLS2.trainable_variables)
# print('*****************************')
# # output=FLS2(train_data_x)
# # print(output)
