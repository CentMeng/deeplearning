# 手写数字识别
## 每个图片是8*8，即64像素，利用每个像素的灰度值来预测数字
### 每个图片8*8
### 识别数字：0，1，2，3，4，5，6，7，8，9

import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix,classification_report #混淆矩阵、分类报告
from sklearn.preprocessing import LabelBinarizer #转化成二维的数据类型，即用0，1来区别，比如
from net.msj.superlearn.classification.neuralnetwork.NeuralNetwork import NeuralNetwork
from sklearn.cross_validation import train_test_split #训练集和测试集区分

digits = load_digits()
X = digits.data
y = digits.target
# 将所有的值转化成0-1之间，最大值减去最小值再除以最大值
X -= X.min()
X /= X.max()

nn = NeuralNetwork([64,100,10],'logistic')#64像素，64个点，结果是0-9 10个数字
X_train,X_test,y_train,y_test = train_test_split(X,y)
labels_train = LabelBinarizer().fit_transform(y_train) #转换成0，1，这是sklearn numpy的要求
labels_test = LabelBinarizer().fit_transform(y_test)
print("start fitting")
nn.fit(X_train,labels_train,epochs=3000) #建立模型
predictions = [] # 预测集合
for i in range(X_test.shape[0]): #测试集测试，循环每一行
    o = nn.predict(X_test[i] )
    print(o)
    predictions.append(np.argmax(o))#取值概率最大的，结果是10个数字，取对应最大的
print(confusion_matrix(y_test,predictions)) #因为结果集是10个数字，所以是10*10矩阵
# [[42  0  0  0  1  0  0  0  0  0] 对角线是预测对的值的个数， 有42个值是0的，预测对了，1代表有1个是数字4（横向第0，1，2，3，4，在4的位置）但预测是0
#  [ 0 27  0  0  0  0  1  0  7  2]
# [ 0  0 43  1  0  0  0  3  2  0]
# [ 0  0  0 37  0  1  0  3  1  0]
# [ 0  0  0  0 55  0  0  1  2  0]
# [ 0  0  0  0  0 41  1  0  0  2]
# [ 1  0  0  0  0  0 40  0  0  0]
# [ 0  0  0  0  0  0  0 52  0  0]
# [ 0  0  0  0  0  1  0  2 36  0]
# [ 0  0  0  2  0  1  0  4  0 38]]
print(classification_report(y_test,predictions))
# precision(预测准确的百分比)
# recall（所有当前数字中，有多少个值预测为此数字的比例，比如）
# f1-score
# support
#      precision   recall   f1-score   support
#
# 0       0.98（结果是0这个值预测准确的百分比是98%）      0.98（结果是0的数字中，有多少个正确的比例）      0.98        43
# 1       1.00      0.73      0.84        37
# 2       1.00      0.88      0.93        49
# 3       0.93      0.88      0.90        42
# 4       0.98      0.95      0.96        58
# 5       0.93      0.93      0.93        44
# 6       0.95      0.98      0.96        41
# 7       0.80      1.00      0.89        52
# 8       0.75      0.92      0.83        39
# 9       0.90      0.84      0.87        45
#
# avg / total       0.92      0.91      0.91       450

