import cv2
import os
import numpy as np
from sklearn import neighbors
import struct
print ("Now start,please wait")

 
def getImages():#处理训练图片
    imgs=np.zeros([60000,784],int)#建立一个60000*784的0矩阵
    for i in range(60000):
        img1=cv2.imread("/home/zhang/Desktop/MNIST/train-images-idx3-ubyte (3)",0)
        for rows in range(28):
            for cols in range(28):#访问每张图片的每个像素，这种方法简单易懂但是效率比较低
                if img1[rows,cols]>=127:#二值化处理，把一整张图片的像素处理成只有0和1
                    img1[rows,cols]=1
                else:
                    img1[rows,cols]=0#这里选择的临界点是127，正好是0-255的中间值
                imgs[i,rows*28+cols]=img1[rows,cols]#把每张图片（28*28）展开成一行（1*784），
                                                    #然后把每张图片的像素逐行放到（60000*784）的大矩阵中
 
    return imgs#返回所有图片的像素重构的矩阵
 
def getLabels():#解析训练标签（解析出来的标签和图片顺序是一一对应的）
    f1=open("/home/zhang/Desktop/MNIST/train-labels-idx1-ubyte (2)",'rb')
    buf1=f1.read()
    f1.close()
    index=0
    magic,num=struct.unpack_from(">II",buf1,0)
    index+=struct.calcsize('>II')
    labs=[]
    labs=struct.unpack_from('>'+str(num)+'B',buf1,index)
    return labs#返回训练标签。之前没有单独解析出来保存在文本文件中，因为解析标签比较简单。
 
def getTestImages():#处理测试图片，和处理训练图片是一样的
    imgs=np.zeros([10000,784],int)#
    for i in range(10000):#
        img1=cv2.imread("/home/zhang/Desktop/MNIST/t10k-images-idx3-ubyte",0)
        for rows in range(28):
            for cols in range(28):
                if img1[rows,cols]>=127:
                    img1[rows,cols]=1
                else:
                    img1[rows,cols]=0
                imgs[i,rows*28+cols]=img1[rows,cols]
    return imgs
 
def getTestLabels():#处理测试标签，和处理训练标签是一样的
    f1=open("/home/zhang/Desktop/MNIST/t10k-labels-idx1-ubyte",'rb')
    buf1=f1.read()
    f1.close()
    index=0
    magic,num=struct.unpack_from(">II",buf1,0)
    index+=struct.calcsize('>II')
    labs=[]
    labs=struct.unpack_from('>'+str(num)+'B',buf1,index)
    return labs
 
if __name__=="__main__":#主函数
 
    print (("Getting train_imgs"))
    train_imgs = getImages()#train_imgs保存60000*784的大矩阵
    print (("Getting train_labels"))
    train_labels = getLabels()#train_labels保存60000个训练标签
    print (("Creating KNN classifier"))
    knn=neighbors.KNeighborsClassifier(algorithm='kd_tree',n_neighbors=3)#重点来了，这里就是加载KNN分类器，具体的用法可以上网搜索
    print (("Training"))
    knn.fit(train_imgs,train_labels)#读入训练图片和标签进行训练
    print (("Getting test_images"))
    test_imgs = getTestImages()#test_imgs保存10000*784的大矩阵
    print (("Getting test_labels"))
    test_labels = getTestLabels()#test_labels保存10000个训练标签
    print (("Predicting"))
    result = knn.predict(test_imgs)#对测试图片进行预测
    wrongNum = np.sum(result!=test_labels)#得出错误个数
    num=len(test_imgs)#训练图片的总数
    print (("Total number:")),num
    print (("Wrong number:")),wrongNum
    print (("RightRate:")),1-wrongNum/float(num)
