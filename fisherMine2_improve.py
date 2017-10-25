# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 14:38:38 2017

@author: Xia
"""
import numpy as np
import pandas as pd

#读取训练数据
def LoadFile(filename):		#load input file containing training data
    filename = 'mb1-3.csv'
    data = pd.read_csv(filename)    
    dataset = data[['Maximumamplitude', 'Mostof','faciescode']]
#    dataset1 = dataset.loc[dataset['faciescode'] == 0]
#    dataset2 = dataset.loc[dataset['faciescode'] == 1]
    return dataset
  
#把数据分类
def ByClass(dataset):		
    classes = np.unique(dataset.values[:,-1])
    div_class = {}
    for i in classes:
        div_class[i] = dataset[dataset.values[:,-1] == i]
    return div_class



#计算每一个属性的均值
def Mean(data):
    mean = data.mean(axis = 0)
    return mean

def transForm(vector, data1, data2):
	mu1 = Mean(np.dot(vector, data1.T))
	mu2 = Mean(np.dot(vector, data2.T))
	return (mu1+mu2)/2, mu1, mu2



#利用训练集类计算投影向量，阀值
def main_train(dataset):		#assuming given two class problem    
    
    dataset_class = ByClass(dataset)
#    print('dataset_class',dataset_class)
    class1, class2 = dataset_class
    class1_data, class2_data = dataset_class[class1], dataset_class[class2] 
#    print(class1_data, class2_data)

    class1_data = dataset1.values[:,:-1]
    class2_data = dataset2.values[:,:-1]
    print(class1_data)
#n1_0  表示行，n1表示列
    n1_0 = class1_data.shape[0]
    n2_0 = class2_data.shape[0]
    n1 = class1_data.shape[1]
    n2 = class2_data.shape[1]
    
    m1 = Mean(class1_data)     #计算第一类的均值
    m2 = Mean(class2_data)      #计算第二类的均值
    m = Mean(dataset.values[:,:-1])      #计算总样本的均值

    #计算类内离散度矩阵
    #第一先变换 X-m 矩阵
    diff1 = class1_data-np.array(list(m1)*n1_0).reshape(n1_0,n1)
    diff2 = class2_data-np.array(list(m2)*n2_0).reshape(n2_0,n2)
    diff = np.concatenate([diff1,diff2])
#    print('diff',diff)
#计算各个样品的类内离散度
    s1 = np.dot(diff1,diff1.T)
    s2 = np.dot(diff2,diff2.T)
    
    m , n = diff.shape
#    print('m,n',m,n)   #314  2
    withinClass = np.zeros((n,n))
    print('withinClass',withinClass)
    #将diff数组对象变化为矩阵
    diff = np.matrix(diff)
#    求Sw,即总类内离散度矩阵
    for i in range(m):
        withinClass += np.dot(diff[i,:].T, diff[i,:])
#    print('withinClass',withinClass)
    #求方向向量
    opt_dir_vector = np.dot(np.linalg.inv(withinClass), (m1 - m2))
    print(opt_dir_vector)
    #将方向向量转化为矩阵，供下面利用
    print ('Vector = ', np.matrix(opt_dir_vector).T)
    
    #将训练集内所有的样品进行投影，可以定义一个专门用于投影的函数
    reflectNum, mu1, mu2 = transForm(opt_dir_vector, class1_data, class2_data)
    print ('reflectNum = ', reflectNum, 'm1 = ', mu1, 'm2 = ', mu2)
    #然后根据公式计算出阀值y0
    y0 = -1/2*(mu1 + mu2)
    return withinClass,y0

if __name__ == '__main__':	
    filename = 'mb1-3.csv'
    dataset = LoadFile(filename)
    main_train(dataset)