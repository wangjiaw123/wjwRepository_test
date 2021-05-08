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



def Gausstype2(Xt_gs,GaussT2_parameter):
    Sigma_gs,M1_gs,M2_gs = GaussT2_parameter[0],GaussT2_parameter[1],\
        GaussT2_parameter[2]
    Sigma_gs=Sigma_gs+0.0001
    m1=tf.minimum(M1_gs,M2_gs)    #m1=<m2
    m2=tf.maximum(M1_gs,M2_gs)
    m_middle=tf.divide(tf.add(m1,m2),2.0)

    if (Xt_gs>=m1) and (Xt_gs<=m_middle):
        mu1=tf.constant(1.0,tf.float32)                
        mu2=tf.exp(-tf.pow(Xt_gs-m2,2.0)/(2.0*tf.pow(Sigma_gs,2.0)))
    elif (Xt_gs>m_middle) and (Xt_gs<=m2):
        mu1=tf.constant(1.0,tf.float32)
        mu2=tf.exp(-tf.pow(Xt_gs-m1,2.0)/(2.0*tf.pow(Sigma_gs,2.0)))
    elif (Xt_gs>m2):
        mu1=tf.exp(-tf.pow(Xt_gs-m2,2.0)/(2.0*tf.pow(Sigma_gs,2.0)))
        mu2=tf.exp(-tf.pow(Xt_gs-m1,2.0)/(2.0*tf.pow(Sigma_gs,2.0)))
    else:
        mu1=tf.exp(-tf.pow(Xt_gs-m1,2.0)/(2.0*tf.pow(Sigma_gs,2.0)))
        mu2=tf.exp(-tf.pow(Xt_gs-m2,2.0)/(2.0*tf.pow(Sigma_gs,2.0))) 

    return mu2,mu1        #mu2<=mu1


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

    def Setting_parameters(self,Grades_set):
        self.FRB_weights.assign(Grades_set[0])
        self.c1.assign(Grades_set[1])
        self.c2.assign(Grades_set[2])
        # for i in range(len(Grades_set[0])):
        #     tf.tensor_scatter_nd_update(self.FRB_weights,tf.constant([[i]]),\
        #         [Grades_set[0][i]])
        # for j in range(len(Grades_set[1])):
        #     tf.tensor_scatter_nd_update(self.c1,tf.constant([[j]]),\
        #         [Grades_set[1][j]])
        #     tf.tensor_scatter_nd_update(self.c2,tf.constant([[j]]),\
        #         [Grades_set[2][j]])

  

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
                    Uu *= mu_big
                    Ll *= mu_small
                UU=tf.tensor_scatter_nd_update(UU,tf.constant([[j]]),[Uu])
                LL=tf.tensor_scatter_nd_update(LL,tf.constant([[j]]),[Ll])
                UU=tf.cast(UU,dtype=tf.float32)
                LL=tf.cast(LL,dtype=tf.float32)
            Output_Left=tf.tensor_scatter_nd_update(Output_Left,tf.constant([[sample_i]]),\
                [self.Compute_LeftPoint(UU,LL)])
            Output_Right=tf.tensor_scatter_nd_update(Output_Right,tf.constant([[sample_i]]),\
                [self.Compute_RightPoint(UU,LL)])
        Output = (Output_Right+Output_Left)/2.0
        #print('Output:',Output)
        return Output

class SingleT2FLS_FWA(tf.keras.Model):
    #构造函数__init__
    def __init__(self,FuzzyRuleNum,FuzzyAntecedentsNum,InitialSetup_List):
        super(SingleT2FLS_FWA,self).__init__()
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
        s1 = tf.reduce_sum(LL_sort)
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
        s1 = tf.reduce_sum(UU_sort)
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

    if len(Xtrain_subMode)%batch_size == 0:
        total_num = len(Xtrain_subMode)//batch_size
    else:
        total_num = len(Xtrain_subMode)//batch_size+1
    
    for Block_id in range(len(Xtrain_subMode) // batch_size):
        with tf.GradientTape() as tape:
            output_data=MODE(Xtrain_subMode[Block_id*batch_size:(Block_id+1)*batch_size,:])
            Loss=lossFunction(output_data,Ytrain_subMode[Block_id*batch_size:(Block_id+1)*batch_size])
        grades=tape.gradient(Loss,MODE.trainable_variables)
        # for i in range(len(grades)):
        #     MODE.trainable_variables[i] = MODE.trainable_variables[i] + learn_rate*grades[i]

        tf.keras.optimizers.Adagrad(learn_rate).apply_gradients(zip(grades,MODE.trainable_variables))
        print('>>>Processes id:{}, Block_id:{}/{},Block_loss:{}'.format(os.getpid(),Block_id+1,total_num,Loss))
        for g_id in range(count):
             subMode_grade[g_id] += grades[g_id]
    #queue.put((subMode_grade,Loss))
    if len(Xtrain_subMode)%batch_size != 0:
        with tf.GradientTape() as tape:
            output_data=MODE(Xtrain_subMode[len(Xtrain_subMode)-(len(Xtrain_subMode) % batch_size):,:])
            Loss=lossFunction(output_data,Ytrain_subMode[len(Xtrain_subMode)-(len(Xtrain_subMode) % batch_size):])
        grades=tape.gradient(Loss,MODE.trainable_variables)
        tf.keras.optimizers.Adagrad(learn_rate).apply_gradients(zip(grades,MODE.trainable_variables))
        print('>>>Processes id:{}, Block_id:{}/{},Block_loss:{}'.format(os.getpid(),Block_id+1,total_num,Loss))
        for g_id in range(count):
            subMode_grade[g_id] += grades[g_id]

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
            for j_num in range(len(q_g[0])):
                Grades_set[0][j_num] = Grades_set[0][j_num] + q_g[0][j_num]
                Grades_set[2][j_num] = Grades_set[2][j_num] + q_g[2][j_num]
            saveloss += q_g[1]
    
        for j_num in range(len(Grades_set[0])):
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

data_Num = 1
Rule = [16]
Epoch_num = 20
processes_num = [processes_N for processes_N in range(12,18,2)]
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
                modeName='Mamdani',modeType=2,predictMode=True,optimizer=tf.keras.optimizers.Adam(0.5),\
                lossFunction=tf.keras.losses.mean_squared_error,batchSIZE=16,epoch=Epoch_num,subMode_learningRate=tf.constant(0.2),processesNum=processes_num[p])
            parallel_time[r,d,p] = _time
            parallel_predict_RMSE[r,d,p] = _predict_RMSE
            parallel_RMSE[r,d,p,:] = _RMSE
            print('r={},d={},p={}'.format(Rule[r],(d+1)*data_size,processes_num[p]))

np.save("parallel_time.npy",parallel_time)
np.save("parallel_predict_RMSE.npy",parallel_predict_RMSE)        
np.save("parallel_RMSE.npy",parallel_RMSE)