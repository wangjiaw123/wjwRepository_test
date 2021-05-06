#/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/5/6
# @Author  : Wangjiawen

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

'''
serial_time = np.load("serial_time.npy")
serial_predict_RMSE = np.load("serial_predict_RMSE.npy")
serial_RMSE = np.load("serial_RMSE.npy")
parallel_time = np.load("parallel_time.npy")
parallel_predict_RMSE = np.load("parallel_predict_RMSE.npy")
parallel_RMSE = np.load("parallel_RMSE.npy")
'''
serial_time = np.random.rand(2,4)
serial_predict_RMSE = np.random.rand(2,4)
serial_RMSE = np.random.rand(2,4,20)
parallel_time = np.random.rand(2,4,8)
parallel_predict_RMSE = np.random.rand(2,4,8)
parallel_RMSE = np.random.rand(2,4,8,20)

'''
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
        plt.grid(True)
        plt.plot(range(1,epoch_NUM+1,1),serial_RMSE[i,j,:])
        plt.xlim(1,epoch_NUM)
        plt.title("Data num:"+str((j+1)*500),fontsize=14)
        plt.ylabel('RMSE',fontsize=14)
        plt.xlabel('epoch',fontsize=14)
        ax1.xaxis.set_major_locator(MultipleLocator(2))
        for k in range(4,17,4):
            plt.plot(range(1,epoch_NUM+1,1),parallel_RMSE[i,j,int(k/2)-1,:])
        plt.legend(["s1","p4","p8","p12","p16"], loc=1)

    


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


plt.tight_layout()
plt.show()













