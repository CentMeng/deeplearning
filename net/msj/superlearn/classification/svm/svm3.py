# 人脸识别问题
## 矩阵的shape方法返回行和列，1是返回行，0是返回列

from __future__ import print_function

from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split # 将集合分为训练集和测试集
from sklearn.datasets import fetch_lfw_people # 引入sklearn中人脸识别的数据集例子
from sklearn.grid_search import GridSearchCV # 循环计算，找出最优方案
from sklearn.metrics import classification_report #分类比较函数，可以比较两个集合
from sklearn.metrics import confusion_matrix # 建立方格
from sklearn.decomposition import RandomizedPCA # 高纬的特征向量降成低纬的特征向量
from sklearn.svm import SVC

print(__doc__)

logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')

lfw_people = fetch_lfw_people(min_faces_per_person=70,resize=0.4) #

n_samples,h,w = lfw_people.images.shape #返回数据集有多少个实例，以及每个实例的h，w

X = lfw_people.data #特征向量矩阵
n_features = X.shape[1] #特征向量的维度，shape返回矩阵的行数和列数，1是返回列数，0是返回行数

y = lfw_people.target #每个实例对应的结果，结果集
target_names = lfw_people.target_names #
n_classes = target_names.shape[0] #返回有多少人进行人脸识别

print("数据集总数：")
print("n_samples:%d"%n_samples);
print("n_features:%d"%n_features);
print("n_classes:%d"%n_classes);

X_train,X_test,y_train,y_test = train_test_split(X,y,0.5)



## 特征值降维
n_components = 150

print("Extracting the top %d eigenfaces from %d faces" %(n_components,X_train.shape[0]))

t0 = time()

pca = RandomizedPCA(n_components=n_components,whiten=True).fit(X_train)
print("1.完成时间：%0.3fs"%(time()-t0))

eigenfaces = pca.components_.reshape((n_components,h,w)) #提取人脸的特征值

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train) #转化成更低纬的特征向量值
X_test_pca = pca.transform(X_test)
print("2.完成时间：%0.3fs"%(time()-t0))
## 特征值降维结束

## 分类器分类
print("分类器分类")
t0 = time()
param_grid = {'C':[1e3,5e3,1e4,5e4,1e5],'gamma':[0.0001,0.0005,0.001,0.005,0.01,0.1],} #C是权重，gamma是特征值中的不同比例，这里C和gamma进行两两组合(这里是5*6，共有30种组合)，通过GridSerachCV找到最佳组合
clf = GridSearchCV(SVC(kernel='rbf',class_weight='auto'),param_grid)
clf = clf.fit(X_train,y_train)
print("分类器分类完成时间：%0.3fs"%(time()-t0))
print("最佳：")
print(clf.best_estimator_)
## 分类器分类完毕

## 测试集预测
print("测试集预测")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("测试预测完成时间：%0.3fs"%(time()-t0))

print("分类报告："+classification_report(y_test,y_pred,target_names=target_names))
print(confusion_matrix(y_test,y_pred,labels=range(n_classes)))
## 测试集预测结束


## 画图函数
def plot_gallery(images,titles,h,w,n_row=3,n_col=4):
    plt.figure(figsize=(1.8*n_col,2.4*n_row))
    plt.subplots_adjust(bottom=0,left=.01,right=.99,top=.99,hspace=.35)
    for i in range(n_row*n_col):
        plt.subplot(n_row,n_col,i+1)
        plt.imshow(images[i].reshape((h,w)),cmap=plt.cm.gray) # 展示图片
        plt.title(titles[i],size=12)
        plt.xticks(());
        plt.yticks(());

def title(y_pred,y_test,target_names,i):
    pred_name = target_names[y_pred[i]].rsplit(' ',1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ',1)[-1]
    return '预测值:%s\n实际值:%s'%(pred_name,true_name)

prediction_titles = [title(y_pred,y_test,target_names,i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test,prediction_titles,h,w)

eigenface_titles = ["eigenface %d" %i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces,eigenface_titles,h,w) #eigenfaces提取部分特征值窗口图像

plt.show()



