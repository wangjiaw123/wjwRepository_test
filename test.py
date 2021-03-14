
import tensorflow as tf
import numpy as np
#from SingleT2FLS_Mamdani import *
#from SingleT2FLS_Mamdani_new import *
from SingleT2FLS_Mamdani_modify import *
import time


N=10
train_data_x=np.random.random((N,6))
train_data_y=np.random.random((N,1))
val_data_x=np.random.random((N,6))
val_data_y=np.random.random((N,1))

trainXY=tf.data.Dataset.from_tensor_slices((train_data_x,train_data_y))
valXY=tf.data.Dataset.from_tensor_slices((val_data_x,val_data_y))


#print(train_data_x)

#print(train_data_y)

LL=[['G','G','G','G','G','G'],['G','G','G','G','G','G'],
    ['G','G','G','G','G','G'],['G','G','G','G','G','G'],
    ['G','G','G','G','G','G'],['G','G','G','G','G','G'],
    ['G','G','G','G','G','G'],['G','G','G','G','G','G'],
    ['G','G','G','G','G','G'],['G','G','G','G','G','G'],
    ['G','G','G','G','G','G'],['G','G','G','G','G','G'],
    ['G','G','G','G','G','G'],['G','G','G','G','G','G'],
    ['G','G','G','G','G','G'],['G','G','G','G','G','G']]


startime=time.time()
FLS2=SingleT2FLS_Mamdani(16,6,LL)

print('FLS2.variables',FLS2.variables)
print('FLS2.trainable_variables:',FLS2.trainable_variables)


#FLS2.build(input_shape=[None,6])
#print(FLS2.summary)
#output_data=FLS2(train_data_x)
#print('output_data',output_data)


# FLS2.compile(optimizer=tf.keras.optimizers.SGD(0.01),\
#     loss=tf.keras.losses.binary_crossentropy,\
#         metrics=['accuracy'])


#FLS2.fit(train_data_x,train_data_y,batch_size=1,epochs=1)
#FLS2.fit(trainXY,epochs=10,validation_data=valXY,validation_freq=1)

    

with tf.GradientTape() as tape:
    output_data=FLS2(train_data_x)
    Loss=tf.norm(output_data-train_data_y,2)/N
grades=tape.gradient(Loss,FLS2.trainable_variables)
print('loss',Loss)


print('grades',grades)




endtime=time.time()
dtime=endtime-startime


print('Totial time:%.8f'%dtime)