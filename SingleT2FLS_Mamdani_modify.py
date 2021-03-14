#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/3/4
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
                  +c1[tf.argmax(c1,0)]-c1[tf.argmin(c1,0)] ,trainable=True)  #初始化c2
        print('***********Initialization of fuzzy rule base parameters completed!*************')
        return FRB_W,FRB_ParaList,c1,c2

    def GetFRB_weights(self):
        return self.FRB_weights,self.c1,self.c2


    def Compute_LeftPoint_locat(self,c1_sort,UU_sort,LL_sort):  
        l_out = 0
        s = 0
        s1 = 0
        b2=c1_sort
        s = tf.reduce_sum(tf.multiply(b2,LL_sort))
        s1 = tf.reduce_sum(LL_sort)
        L_out = tf.ones(self.Rule_num)
        l_out = s/s1
        for i in range(self.Rule_num):
            s += b2[i]*(UU_sort[i]-LL_sort[i])
            s1 += UU_sort[i]-LL_sort[i]
            l_out = tf.minimum(l_out,s/s1)
            L_out = tf.tensor_scatter_nd_update(L_out,tf.constant([[i]]),[tf.minimum(l_out,s/s1)])
        L_locat = tf.argmin(L_out,0)
        return L_locat

    def Compute_RightPoint_locat(self,c2_sort,UU_sort,LL_sort):
        tf.keras.backend.set_floatx('float32')
        r_out = 0
        s = 0
        s1 = 0
        b1=c2_sort
        s = tf.reduce_sum(tf.multiply(b1,UU_sort))
        s1 = tf.reduce_sum(UU_sort)
        R_out = tf.ones(self.Rule_num)
        r_out = s/s1
        for i in range(self.Rule_num):
            s += b1[i]*(LL_sort[i]-UU_sort[i])
            s1 += LL_sort[i]-UU_sort[i]
            r_out = tf.maximum(r_out,s/s1)
            R_out = tf.tensor_scatter_nd_update(R_out,tf.constant([[i]]),[tf.maximum(r_out,s/s1)])

        R_locat = tf.argmax(R_out)

        return R_locat

    def call(self,input_data):
        #tf.keras.backend.set_floatx('float64')
        samples_num = input_data.shape[0]
        #Output_Left=tf.Variable(tf.zeros((samples_num,)))
        #Output_Right=tf.Variable(tf.zeros((samples_num,)))

        Output_Left=tf.ones(samples_num)
        Output_Right=tf.ones(samples_num)
        ###########################################
        for sample_i in range(samples_num):
            ###########################################
        
            input = input_data[sample_i]
            print('**///////////////********input_data',input)
                #UU=tf.Variable(tf.zeros((self.Rule_num,)))    #tf.random.get_global_generator().normal(shape=(self.Rule_num,))
                #LL=tf.Variable(tf.zeros((self.Rule_num,)))    #tf.random.get_global_generator().normal(shape=(self.Rule_num,))
                # UU=np.ones(self.Rule_num)
                # LL=np.ones(self.Rule_num)
            UU = tf.ones(self.Rule_num)
            LL = tf.ones(self.Rule_num)
            for j in range(self.Rule_num):
                Uu = tf.constant(1.0)
                Ll = tf.constant(1.0)
                #print('Uu={0},Ll={1}'.format(Uu,Ll))
                for k in range(self.Antecedents_num):
                    locat_num = self.Antecedents_num*j+k
                    if self.Init_SetupList[j][k]=='G':
                        #print('----------input[0]',input[0])
                        mu_small,mu_big = Gausstype2(input[k],self.FRB_weights[locat_num:locat_num+3])
                    #print('Uu:',Uu) 
                    #print('mu_big:',mu_big)   
                    Uu = Uu*mu_big
                    Ll = Ll*mu_small
                    #print('*-*-*-*-*-*-*-*-*-*--*-*-*-*-')
                    #print('Uu:',Uu)
                    #print('Ll:',Ll)
                    #print('mu_small:',mu_small)
                    # UU[j].assign(Uu)
                    # LL[j].assign(Ll)
                    #print('Uu:',Uu)
                    #print('Uu.shape:',tf.shape(Uu))
                UU=tf.tensor_scatter_nd_update(UU,tf.constant([[j]]),[Uu])
                LL=tf.tensor_scatter_nd_update(LL,tf.constant([[j]]),[Ll])
                    #print('UU[j]-Uu:',UU[j]-Uu)
                    #print('UU[j].shape:',tf.shape(UU[j]))
            #print('---------------------------------------------')
            #print('UU:',UU)
            #print('LL:',LL)
            #tf.cast(UU,dtype=tf.float32)
            #tf.cast(LL,dtype=tf.float32)
            L_c1_sort = tf.sort(self.c1,direction='ASCENDING')
            L_c1_index = tf.argsort(self.c1,direction='ASCENDING')
            L_UU_sort = tf.gather(UU,L_c1_index)
            L_LL_sort = tf.gather(LL,L_c1_index)  
            #print('L_UU_sort:',L_UU_sort)
            #print('*********L_LL_sort= \n',L_LL_sort)            

            R_c2_sort = tf.sort(self.c2,direction='ASCENDING')
            R_c2_index = tf.argsort(self.c2,direction='ASCENDING')
            R_UU_sort = tf.gather(UU,R_c2_index)
            R_LL_sort = tf.gather(LL,R_c2_index)
            #print('R_UU_sort:',R_UU_sort)
            #print('*********R_LL_sort= \n',R_LL_sort)

            L_locat = self.Compute_LeftPoint_locat(L_c1_sort,L_UU_sort,L_LL_sort)
            R_locat = self.Compute_RightPoint_locat(R_c2_sort,R_UU_sort,R_LL_sort)
            #print('***********L_locat= \n',L_locat)
            #print('*********************R_locat= \n',R_locat)
            #print('self.count:',self.count)
            if self.count==0:
                tf.no_gradient('self.Compute_LeftPoint_locat')
                tf.no_gradient('self.Compute_RightPoint_locat')
            self.count+=1

            #print('L_UU_sort:',L_UU_sort)
            #print('*********L_LL_sort= \n',L_LL_sort)   
            #print('R_UU_sort:',R_UU_sort)
            #print('*********R_LL_sort= \n',R_LL_sort)

            #index=tf.constant([[x] for x in range(self.Rule_num)])

            #print('**---------++++++++++++++------**********--------++++++++')

            L_U_sort = tf.ones(self.Rule_num)
            R_L_sort = tf.ones(self.Rule_num)
            for i in range(L_locat+1):
                L_U_sort=tf.tensor_scatter_nd_update(L_U_sort,tf.constant([[i]]),[L_UU_sort[i]])
            for j in range(L_locat+1,self.Rule_num,1):
                L_U_sort=tf.tensor_scatter_nd_update(L_U_sort,tf.constant([[j]]),[L_LL_sort[j]])

            #L_U_sort[0:L_locat+1].assign(L_UU_sort[0:L_locat+1],tf.float32)
            #print('-+-+-+-+-+-L_U_sort-*-**-*-*-*-*++*-*-/-///-**+-+-+:',L_U_sort)

            #L_U_sort[L_locat+1:].assign(L_LL_sort[L_locat+1:],tf.float32)
            # L_U_sort[0:L_locat+1] = tf.tensor_scatter_nd_update(L_U_sort[0:L_locat+1],\
            #     tf.constanf([[x] for x in range(L_locat+1)],[[y] for y in L_UU_sort[0:L_locat+1]])
            # L_U_sort[L_locat+1:] = tf.tensor_scatter_nd_update(L_U_sort[L_locat+1:],\
            #     tf.constanf([[x] for x in range(L_locat+1:self.Rule_num)],[[y] for y in L_LL_sort[L_locat+1:]])
            for i in range(R_locat+1):
                R_L_sort=tf.tensor_scatter_nd_update(R_L_sort,tf.constant([[i]]),[R_LL_sort[i]])
            for j in range(R_locat+1,self.Rule_num,1):
                R_L_sort=tf.tensor_scatter_nd_update(R_L_sort,tf.constant([[j]]),[R_UU_sort[j]])  

            #print('-+-+-+-+-+-R_L_sort-*-**-*-*-*-*++*-*-/-///-**+-+-+:',R_L_sort)
            #R_L_sort[0:R_locat+1].assign(R_LL_sort[0:R_locat+1])
            #R_L_sort[R_locat+1:].assign(R_UU_sort[R_locat+1:])
            #Output_Left = tf.Variable(1.0)

            Output_Left=tf.tensor_scatter_nd_update(Output_Left,tf.constant([[sample_i]]),\
                [tf.reduce_sum(tf.multiply(self.c1,L_U_sort))/tf.reduce_sum(L_U_sort)])
            #print('Output_left:',Output_Left)
            Output_Right=tf.tensor_scatter_nd_update(Output_Right,tf.constant([[sample_i]]),\
                [tf.reduce_sum(tf.multiply(self.c2,R_L_sort))/tf.reduce_sum(R_L_sort)])

            Output = (Output_Right+Output_Left)/2.0

            #print('single_t2fls__mamdani.trainable_variables:',single_t2fls__mamdani.trainable_variables)

            #print('Output:',Output)
        return Output


















'''
        all_weight = list()
        for i in range(self.Rule_num):
            all_weight.append(dict())
            for j in range(self.Antecedents_num):
                if InitialSetup_List[i][j] == 'G':
                   key = 'w{}'.format(i*self.Antecedents_num+j+1)
                   all_weight[i][key] = tf.Variable(tf.abs(tf.random.get_global_generator().normal(shape=(3,)))) +0.1  #初始化模糊规则库中隶属函数的参数
        c1 = tf.Variable(tf.abs(tf.random.get_global_generator().normal(shape=(self.Rule_num,))))                    #初始化c1
        c2 = tf.Variable(tf.abs(tf.random.get_global_generator().normal(shape=(self.Rule_num,))))+c1[tf.argmax(c1,0)]-c1[tf.argmin(c1,0)]   #初始化c2
        
        print('***********Initialization of fuzzy rule base parameters completed!*************')
        #print(all_weight)

        return all_weight,c1,c2
'''







'''
LL=[['G','G','G','G','G','G'],['G','G','G','G','G','G'],['G','G','G','G','G','G'],['G','G','G','G','G','G']]

FLS2=SingleT2FLS(4,6,LL,Mam)

print(FLS2.GetFRB_weights())

'''








'''
        for i in range(samples_num):
            input = input_data[i]
            #UU=tf.Variable(tf.zeros((self.Rule_num,)))    #tf.random.get_global_generator().normal(shape=(self.Rule_num,))
            #LL=tf.Variable(tf.zeros((self.Rule_num,)))    #tf.random.get_global_generator().normal(shape=(self.Rule_num,))
            # UU=np.ones(self.Rule_num)
            # LL=np.ones(self.Rule_num)
            UU = np.ones(self.Rule_num)
            LL = np.ones(self.Rule_num)
            for j in range(self.Rule_num):
                Uu = tf.ones(1,tf.float32) 
                Ll = tf.ones(1,tf.float32)
                for k in range(self.Antecedents_num):
                    locat_num = self.Antecedents_num*j+k

                    if self.Init_SetupList[j][k]=='G':
                        mu_small,mu_big = Gausstype2(input[k],self.FRB_weights[locat_num:locat_num+3])

                    Uu *= mu_big
                    Ll *= mu_small
                # UU[j].assign(Uu)
                # LL[j].assign(Ll)
                #print('Uu:',Uu)
                #print('Uu.shape:',tf.shape(Uu))
                UU[j]=Uu[0]
                LL[j]=Ll[0]
                #print('UU[j]-Uu:',UU[j]-Uu)
                #print('UU[j].shape:',tf.shape(UU[j]))

            tf.cast(UU,dtype=tf.float32)
            tf.cast(LL,dtype=tf.float32)
            L_c1_sort = tf.sort(self.c1,direction='ASCENDING')
            L_c1_index = tf.argsort(self.c1,direction='ASCENDING')
            L_UU_sort = tf.gather(UU,L_c1_index)
            L_LL_sort = tf.gather(LL,L_c1_index)  
            print('L_UU_sort:',L_UU_sort)
            print('*********L_LL_sort= \n',L_LL_sort)            

            R_c2_sort = tf.sort(self.c2,direction='ASCENDING')
            R_c2_index = tf.argsort(self.c2,direction='ASCENDING')
            R_UU_sort = tf.gather(UU,R_c2_index)
            R_LL_sort = tf.gather(LL,R_c2_index)
            print('R_UU_sort:',R_UU_sort)
            print('*********R_LL_sort= \n',R_LL_sort)

            # L_locat = self.Compute_LeftPoint_locat(L_c1_sort,L_UU_sort,L_LL_sort)
            # R_locat = self.Compute_RightPoint_locat(R_c2_sort,R_UU_sort,R_LL_sort)
            L_locat = self.Compute_LeftPoint_locat(tf.cast(L_c1_sort,dtype=tf.float32),tf.cast(L_UU_sort,dtype=tf.float32),tf.cast(L_LL_sort,dtype=tf.float32))
            R_locat = self.Compute_RightPoint_locat(tf.cast(R_c2_sort,dtype=tf.float32),tf.cast(R_UU_sort,dtype=tf.float32),tf.cast(R_LL_sort,dtype=tf.float32))
            print('***********L_locat= \n',L_locat)
            print('*********************R_locat= \n',R_locat)
            if i==0:
                tf.no_gradient('self.Compute_LeftPoint_locat')
                tf.no_gradient('self.Compute_RightPoint_locat')

            print('L_UU_sort:',L_UU_sort)
            print('*********L_LL_sort= \n',L_LL_sort)   
            print('R_UU_sort:',R_UU_sort)
            print('*********R_LL_sort= \n',R_LL_sort)


            # L_UU_sort[L_locat:] = L_LL_sort[L_locat:]
            # R_LL_sort[R_locat:] = R_UU_sort[R_locat:]
            L_U_sort = np.ones(self.Rule_num)
            R_L_sort = np.ones(self.Rule_num)
            L_U_sort[0:L_locat+1] = L_UU_sort[0:L_locat+1]
            L_U_sort[L_locat+1:] = L_LL_sort[L_locat+1:]

            R_L_sort[0:L_locat+1] = R_LL_sort[0:L_locat+1]
            R_L_sort[L_locat+1:] = R_UU_sort[L_locat+1:]
            # L_UU_sort[L_locat:].assign(L_LL_sort[L_locat:],dtype=tf.float32)
            # R_LL_sort[R_locat:].assign(R_UU_sort[R_locat:],dtype=tf.float32)



            # Output_Left[i] = tf.reduce_sum(tf.multiply(tf.cast(self.c1,dtype=tf.float32),\
            #     tf.cast(L_U_sort,dtype=tf.float32)))/tf.reduce_sum(tf.cast(L_U_sort,dtype=tf.float32))
            # Output_Right[i] = tf.reduce_sum(tf.multiply(tf.cast(self.c2,dtype=tf.float32),\
            #     tf.cast(R_L_sort,dtype=tf.float32)))/tf.reduce_sum(tf.cast(R_L_sort,dtype=tf.float32))

            # Output_Left[i] = self.compute_FLSoutput(self.c1,L_U_sort)
            # Output_Right[i] = self.compute_FLSoutput(self.c2,R_L_sort)

            #Output_Left[i].assign(self.Compute_LeftPoint(UU,LL))
            #Output_Right[i].assign(self.Compute_RightPoint(UU,LL))
            #Output_Left[i]=self.Compute_LeftPoint(tf.cast(UU,dtype=tf.float32),tf.cast(LL,dtype=tf.float32))
            #Output_Right[i]=self.Compute_RightPoint(tf.cast(UU,dtype=tf.float32),tf.cast(LL,dtype=tf.float32))
        #Output = tf.divide(tf.add(Output_Left,Output_Right),2.0)
'''