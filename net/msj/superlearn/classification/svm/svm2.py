import numpy as np
import pylab as pl
from sklearn import svm

# 创建40个点
np.random.seed(0)
X = np.r_[np.random.randn(20,2)-[2,2],np.random.randn(20,2)+[2,2]] #[np.random.randn(20,2)-[2,2]
# randn(20,2)生成20行，2维的正态分布数 -[2,2]是减去正态分布的左侧，+[2,2]是为了增加正态分布使在右侧
Y = [0] * 20 + [1] *20 #一行数据有20个是0类的20个1类的

clf = svm.SVC(kernel="linear")
clf.fit(X,Y)

# 获取超平面
w = clf.coef_[0] # 获取系数
k = -w[0]/w[1] #斜率
xx = np.linspace(-5,5) # 从-5到5之间产生几个连续的值
yy = k*xx - (clf.intercept_[0])/w[1] #INTERCEPT是一个函数，指函数图形与坐标交点到原点的距离


# 获取支持向量相切的线
b = clf.support_vectors_[0]
yy_down = k*xx + (b[1]-k*b[0])
b = clf.support_vectors_[-1]
yy_up = k*xx + (b[1]-k*b[0])

print("w:",w)
print("k:",k)
print("xx",xx)
print("support_vector",clf.support_vectors_)
print("coef_",clf.coef_)
print("yy",xx)
print("yy_down",xx)
print("yy_up",xx)

# 画图
pl.plot(xx,yy,"k-")
pl.plot(xx,yy_down,"k--")
pl.plot(xx,yy_up,"k--")

pl.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],s=80,facecolors="none") #单独圈出来支持向量点

pl.scatter(X[:,0],X[:,1],c=Y,cmap=pl.cm.Paired) # 画出所有点

pl.axis("tight")
pl.show()