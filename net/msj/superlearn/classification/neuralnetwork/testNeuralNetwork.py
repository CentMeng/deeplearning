#测试神经网络算法
## 简单的测试神经网络算法
### 简单非线性关系数据集测试： 测试结果，异或的输出
#### X:          Y
#### 0,0         0
#### 0,1         1
#### 1,0         1
#### 1,1         0
from net.msj.superlearn.classification.neuralnetwork.NeuralNetwork import NeuralNetwork

import numpy as np

nn = NeuralNetwork([2,2,1],'tanh')#第一层输入层是二维的，所以有两个神经元，第二层是隐藏层也设置2个神经元，第三层输出层是1个神经元
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])
nn.fit(X,y)
for i in [[0,0],[0,1],[1,0],[1,1]]:
    print(i,nn.predict(i))

#某次测试结果，可以看出测试答案和结果相同，趋近与0或1
## [0, 0] [-0.00053143]
## [0, 1] [ 0.99838732]
## [1, 0] [ 0.99828494]
## [1, 1] [ 0.01188015]


