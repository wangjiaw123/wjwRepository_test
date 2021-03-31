
import time
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
#from SingleT2FLS_Mamdani import *
# from SingleT2FLS_TSK import *
from SingleT2FLS_FWA import *
from tensorflow.python.keras.backend import arange
#from tensorflow.python.ops.array_ops import zeros
#from SingleT2FLS_Mamdani_new import *
#from SingleT2FLS_Mamdani_modify import *

#model_path='E:\FLS_Tensorflow\新版本\最新版\\mode_FLS.h5'


# gpus = tf.config.list_physical_devices("GPU")
# print(gpus)
# if gpus:
#     gpu0 = gpus[0] #如果有多个GPU，仅使用第0个GPU
#     tf.config.experimental.set_memory_growth(gpu0, True) #设置GPU显存用量按需使用
#     # 或者也可以设置GPU显存为固定使用量(例如：4G)
#     #tf.config.experimental.set_virtual_device_configuration(gpu0,
#     #    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]) 
#     tf.config.set_visible_devices([gpu0],"GPU")


# N=100
# train_data_x=np.random.random((N,6))
# train_data_y=np.random.random((N,1))
# val_data_x=np.random.random((N,6))
# val_data_y=np.random.random((N,1))
#trainXY=tf.data.Dataset.from_tensor_slices((train_data_x,train_data_y))
#valXY=tf.data.Dataset.from_tensor_slices((val_data_x,val_data_y))


def DF(x):
    a=0.2
    return (a*x)/(1+x**10)

def Mackey_Glass(N,tau):
    t=np.zeros(N)
    x=np.zeros(N)
    x[0],t[0]=1.2,0
    b,h=0.1,0.1
    for k in arange(N-1):
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
for i in arange(n_train):
    x_star[i]=y[i+10]
#plt.plot(arange(1,n_train+1,1),x_star)
#plt.show()



AntecedentsNum=4
data_size=200
multiple=1
X_train=np.zeros([multiple*data_size-3,AntecedentsNum])
Y_train=np.zeros(multiple*data_size-3)
X_test=np.zeros([data_size-3,AntecedentsNum])
Y_test=np.zeros(data_size-3)

X_train[:,0]=x_star[0:multiple*data_size-3]
X_train[:,1]=x_star[1:multiple*data_size-2]
X_train[:,2]=x_star[2:multiple*data_size-1]
X_train[:,3]=x_star[3:multiple*data_size]
Y_train=x_star[4:multiple*data_size+1]

X_test[:,0]=x_star[multiple*data_size+1:(multiple+1)*data_size-2]
X_test[:,1]=x_star[multiple*data_size+2:(multiple+1)*data_size-1]
X_test[:,2]=x_star[multiple*data_size+3:(multiple+1)*data_size]
X_test[:,3]=x_star[multiple*data_size+4:(multiple+1)*data_size+1]
Y_test=x_star[multiple*data_size+5:(multiple+1)*data_size+2]


LL=[['G','G','G','G','G','G'],['G','G','G','G','G','G'],
    ['G','G','G','G','G','G'],['G','G','G','G','G','G'],
    ['G','G','G','G','G','G'],['G','G','G','G','G','G'],
    ['G','G','G','G','G','G'],['G','G','G','G','G','G'],  
    ['G','G','G','G','G','G'],['G','G','G','G','G','G'],
    ['G','G','G','G','G','G'],['G','G','G','G','G','G'],
    ['G','G','G','G','G','G'],['G','G','G','G','G','G'],
    ['G','G','G','G','G','G'],['G','G','G','G','G','G']]


startime=time.time()
# FLS2=SingleT2FLS_Mamdani(16,4,LL)
# FLS2=SingleT2FLS_TSK(8,4,LL)
FLS2=SingleT2FLS_FWA(8,4,LL)
print('***********************************')
print('FLS2.variables',FLS2.variables)
print('***********************************')
print('FLS2.trainable_variables:',FLS2.trainable_variables)
print('***********************************')


#FLS2.build(input_shape=[None,6])
#print(FLS2.summary)
#output_data=FLS2(train_data_x)
#print('output_data',output_data)


#FLS2.compile(optimizer=tf.keras.optimizers.SGD(0.01),\
#     loss=tf.keras.losses.binary_crossentropy,\
#         metrics=['accuracy'])


#FLS2.fit(train_data_x,train_data_y,batch_size=1,epochs=1)
#FLS2.fit(trainXY,epochs=10,validation_data=valXY,validation_freq=1)

optimizer=tf.keras.optimizers.Adam(0.5)  # (Mamdani-SGD-1.5,TSK-Adam-1)
mse=tf.keras.losses.mean_squared_error    

epoch = 20
Loss_save = np.zeros(epoch)

for epo in arange(epoch):
    with tf.GradientTape() as tape:
        output_data=FLS2(X_train)
        Loss=mse(output_data,Y_train)#/len(Y_train)
        print('epoch:{0},Loss:{1}'.format(epo+1,Loss))
    grades=tape.gradient(Loss,FLS2.trainable_variables)
    print('grades:',grades)
    #print('grades',grades)
    #FLS2_trainVariable=FLS2.trainable_variables
    #print('FLS2_trainVariable_old:',FLS2_trainVariable)
    #print('loss',Loss)
    optimizer.apply_gradients(zip(grades,FLS2.trainable_variables))
    #print('FLS2_trainVariable_new:',FLS2.trainable_variables)
    Loss_save[epo]=tf.reduce_sum(Loss).numpy()
    #print('FLS2_trainVariable_new:',FLS2.trainable_variables[0]-FLS2_trainVariable[0])
#model.save(filepath=model_path)

plt.subplot(1,3,1)
plt.plot(range(1,epoch+1,1),Loss_save)

plt.subplot(1,3,2)
YY_fls_out_train=FLS2(X_train)
plt.plot(range(1,len(Y_train)+1,1),Y_train)
plt.plot(range(1,len(Y_train)+1,1),YY_fls_out_train)

YY_fls_out_test=FLS2(X_test)
plt.subplot(1,3,3)
plt.plot(range(1,len(Y_test)+1,1),Y_test)
plt.plot(range(1,len(Y_test)+1,1),YY_fls_out_test)



endtime=time.time()
dtime=endtime-startime
print('Totial time:%.8f'%dtime)


plt.show()

#FLS2.save_weights('FLS2_weights.ckpt')    #保存模型参数到FLS2_weights.ckpt
#print("*****Saved total model!******")


#del FLS2

#FLS2=SingleT2FLS_Mamdani(16,4,LL)
#FLS2.load_weights('FLS2_weights.ckpt')   #恢复模型
#print('**********Test**********',FLS2(X_train))

