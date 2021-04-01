#/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/3/31
# @Author  : Wangjiawen
import tensorflow as tf

def Gauss1mf(x,GaussT1_parameter):  #高斯型隶属函数,2个参数
    sigma,mu = GaussT1_parameter[0],GaussT1_parameter[1]
    sigma+=0.0001
    return tf.exp(-tf.pow(x-mu,2.0)/(2.0*tf.pow(sigma,2.0)))

def Gauss2mf(x,Gauss2mfT1_parameter): #双边高斯隶属函数,4个参数
    sigma1,sigma2 = Gauss2mfT1_parameter[0],Gauss2mfT1_parameter[2]
    c1=tf.minimum(Gauss2mfT1_parameter[1],Gauss2mfT1_parameter[3])
    c2=tf.maximum(Gauss2mfT1_parameter[1],Gauss2mfT1_parameter[3])
    if x<=c1:
        return tf.exp(-tf.pow(x-c1,2.0)/(2.0*tf.pow(sigma1,2.0)))
    else:
        return tf.exp(-tf.pow(x-c2,2.0)/(2.0*tf.pow(sigma2,2.0)))

def Trapmf(x,TrapmfT1_parameter):     #梯形隶属函数,4个参数
    TrapmfT1_parameter = tf.sort(TrapmfT1_parameter,direction='ASCENDING')
    a,b = TrapmfT1_parameter[0],TrapmfT1_parameter[1]
    c,d = TrapmfT1_parameter[2],TrapmfT1_parameter[3]

    if a<x<=b:
        return (x-a)/(b-a)
    elif b<x<=c:
        return 1
    elif c<x<=d:
        return (d-x)/(d-c)
    else:
        return 0
    
def Trimf(x,TrimfT1_parameter):       #三角形隶属函数,3个参数
    a,b,c = TrimfT1_parameter[0],TrimfT1_parameter[1],TrimfT1_parameter[2]
    if a<=x<=b:
        return (x-a)/(b-a)
    elif b<x<=c:
        return (c-x)/(c-b)
    else:
        return 0

def Sigmf(x,SigmfT1_parameter):       #Sigmf隶属函数,2个参数
    a,c=SigmfT1_parameter[0],SigmfT1_parameter[1]
    return 1/(1+tf.exp(-a*(x-c)))
    

def Gbellmf(x,GbellmfT1_parameter):   #钟型隶属函数 ,3个参数
    a,b,c = GbellmfT1_parameter[0],GbellmfT1_parameter[1],GbellmfT1_parameter[2]
    return 1/(1+tf.pow(tf.abs((x-c)/a),2*b))


def Psigmf(x,PsigmfT1_parameter):     # 4个参数
    return Sigmf(x,PsigmfT1_parameter[0:2])*Sigmf(x,PsigmfT1_parameter[2:4])

def Dsigmf(x,PsigmfT1_parameter):     # 4个参数
    return tf.abs(Sigmf(x,PsigmfT1_parameter[0:2])-Sigmf(x,PsigmfT1_parameter[2:4]))

