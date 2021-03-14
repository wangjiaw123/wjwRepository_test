


import tensorflow as tf

def Gausstype2(Xt_gs,GaussT2_parameter):
    Sigma_gs,M1_gs,M2_gs = GaussT2_parameter[0],GaussT2_parameter[1],\
        GaussT2_parameter[2]
    Sigma_gs=tf.abs(Sigma_gs)+0.00000001
    m1=tf.minimum(M1_gs,M2_gs)    #m1=<m2
    m2=tf.maximum(M1_gs,M2_gs)
    m_middle=tf.divide(tf.add(m1,m2),2)

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

'''
import time
import numpy as np
startime=time.time()
print('***********')   
for i in range(10000): 
    a=np.random.random((3,))
    b=5*np.random.random((1,))
    print(Gausstype2(b,a))
endtime=time.time()
dtime=endtime-startime
print('Totial time:%.8f'%dtime)
'''
