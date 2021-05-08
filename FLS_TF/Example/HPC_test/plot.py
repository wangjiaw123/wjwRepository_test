#/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/5/6
# @Author  : Wangjiawen

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


serial_time = np.load("serial_time.npy")
serial_predict_RMSE = np.load("serial_predict_RMSE.npy")
serial_RMSE = np.load("serial_RMSE.npy")
parallel_time = np.load("parallel_time.npy")
parallel_predict_RMSE = np.load("parallel_predict_RMSE.npy")
parallel_RMSE = np.load("parallel_RMSE.npy")
'''
serial_time = np.random.rand(1,4)
serial_predict_RMSE = np.random.rand(1,4)
serial_RMSE = np.random.rand(1,4,20)
parallel_time = np.random.rand(1,4,8)
parallel_predict_RMSE = np.random.rand(1,4,8)
parallel_RMSE = np.random.rand(1,4,8,20)



time_now = time.strftime("%Y%m%d-%H%M",time.localtime())
#SavePath = os.path.split(__file__)[0]+'/'+time_now
SavePath = os.getcwd()+'/'+time_now+"Save_picture"
if not os.path.exists(SavePath):
    os.makedirs(SavePath)
    print('*************Picture save path:{}*************'.format(SavePath))
'''

Rule_NUM,data_NUM,processer_NUM,epoch_NUM = list(np.shape(parallel_RMSE))
speedup_ratio = np.zeros([Rule_NUM,data_NUM,processer_NUM])

for i in range(Rule_NUM):
    for j in range(data_NUM):
        for k in range(processer_NUM):
            speedup_ratio[i,j,k] = serial_time[i,j]/parallel_time[i,j,k]
print(speedup_ratio)


for i in range(Rule_NUM):
    plt.figure()
    for j in range(data_NUM):
        ax1 = plt.subplot(2,2,j+1)
        #plt.grid(True)
        
        plt.xlim(1,epoch_NUM)
        plt.title("Train data num:"+str((j+1)*500),fontsize=14)
        plt.ylabel('RMSE',fontsize=14)
        plt.xlabel('epoch',fontsize=14)
        ax1.xaxis.set_major_locator(MultipleLocator(2))

        #for k in range(4,17,4):
            #plt.plot(range(1,epoch_NUM+1,1),parallel_RMSE[i,j,int(k/2)-1,:])
        #plt.legend(["s1","p4","p8","p12","p16"], loc=1)

        for k in range(0,8,):
            plt.plot(range(1,epoch_NUM+1,1),parallel_RMSE[i,j,k,:])

        plt.plot(range(1,epoch_NUM+1,1),serial_RMSE[i,j,:],linewidth='2',color='black')
        plt.legend(["s1","p2","p4","p6","p8","p10","p12","p14","p16"],loc=1)



plt.figure()
ax1_1 = plt.subplot(1,2,1)
#plt.grid(True)
plt.xlim(2,16)
plt.ylabel('Speedup Ratio',fontsize=14)
plt.xlabel('Processer num',fontsize=14)
ax1_1.xaxis.set_major_locator(MultipleLocator(2))
for j in range(data_NUM):
    plt.plot(range(2,18,2),speedup_ratio[0,j,:])
    plt.scatter(range(2,18,2),speedup_ratio[0,j,:])
plt.legend(["d500","d1000","d1500","d2000"])

ah=[]
for j in range(4):
    ah.append(speedup_ratio[0,j,j])
ax1_2 = plt.subplot(1,2,2)
plt.xlim(4,16)
plt.ylabel('Speedup Ratio',fontsize=14)
plt.xlabel('Processer num',fontsize=14)    
ax1_2.xaxis.set_major_locator(MultipleLocator(4))
plt.plot(range(4,18,4),ah)
plt.scatter(range(4,18,4),ah)




for i in range(Rule_NUM):
    ax1_3=plt.figure()
    ax=plt.gca()
    plt.plot(range(500,4*500+1,500),serial_predict_RMSE[i,:],linewidth='2',color='black')
    plt.scatter(range(500,4*500+1,500),serial_predict_RMSE[i,:],linewidth='2',color='black')
    plt.xlim(450,2050)
    plt.title("Predict RMSE",fontsize=14)
    plt.ylabel('RMSE',fontsize=14)
    #plt.grid(True)
    plt.xlabel('Data num',fontsize=14)
    ax.xaxis.set_major_locator(MultipleLocator(500))
    for j in range(processer_NUM):
        plt.plot(range(500,4*500+1,500),parallel_predict_RMSE[i,:,j])
        plt.scatter(range(500,4*500+1,500),parallel_predict_RMSE[i,:,j])
    #plt.legend(["s1","p4","p8","p12","p16"], loc=1)
    plt.legend(["s1","p2","p4","p6","p8","p10","p12","p14","p16"], loc=2,bbox_to_anchor=(1.05,1))




plt.tight_layout()
plt.show()


'''
plt.figure()
for i in range(Rule_NUM):
    ax1_1 = plt.subplot(1,2,i+1)
    plt.grid(True)
    plt.xlim(2,16)
    plt.ylabel('Speedup Ratio',fontsize=14)
    plt.xlabel('Processer num',fontsize=14)
    ax1_1.xaxis.set_major_locator(MultipleLocator(2))
    for j in range(data_NUM):
        plt.plot(range(2,18,2),speedup_ratio[i,j,:])



ah=[[],[]]
for j in range(4):
    ah[0].append(speedup_ratio[0,j,2*j-1])
    ah[1].append(speedup_ratio[1,j,2*j-1])

plt.figure()
for i in range(Rule_NUM):
    ax1_2 = plt.subplot(1,2,i+1)
    plt.grid(True)
    plt.xlim(4,16)
    plt.ylabel('Speedup Ratio',fontsize=14)
    plt.xlabel('Processer num',fontsize=14)    
    ax1_2.xaxis.set_major_locator(MultipleLocator(4))
    plt.plot(range(4,18,4),ah[i])
'''














