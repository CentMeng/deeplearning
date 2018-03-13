#神经网络算法实现
#双曲正切函数和逻辑函数都是s函数
import numpy as np

def tanh(x): #定义双曲正切函数
    return np.tanh(x)

def tanh_deriv(x): #定义双曲正切函数的导数
    return 1.0 - np.tanh(x)*np.tanh(x)

def logistic(x): #逻辑函数
    return  1/(1 + np.exp(-x))

def logistic_derivative(x): #对逻辑函数求导
    return logistic(x)*(1-logistic(x))


class NeuralNetwork:
    def __init__(self,layers,activation='tanh'):  #两个下划线+init+两个下划线是构造函数，此处设计了默认值tanh
        #self 相当于java的this
        #layers python里面的list，每层里面有多少个神经元
        #activation 指定的模式
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv

        self.weights = []
        #初始化权重
        for i in range(1,len(layers)-1): # len(layers)得到有几层
            self.weights.append((2*np.random.random((layers[i-1]+1,layers[i]+1))-1)*0.25)
            self.weights.append((2*np.random.random((layers[i]+1,layers[i+1]))-1)*0.25)


    #正向传输和反向传输，根据算法
    def fit(self,X,y,learning_rate=0.2,epochs=10000):# X二维数据每一行对应有多少个实例，每一行中的每一列对应有多少特征值，y对应每个实例结果集，每次随机抽取epochs个实例（抽样方式），learning_rate是梯度下降算法的下坡步伐值
        X=np.atleast_2d(X) #确定是否是2D数组
        temp=np.ones([X.shape[0],X.shape[1]+1]) #初始化矩阵，ones都是1，参数是几行几列
        temp[:,0:-1]=X # :是取所有，0:-1是指从第一列到除了最后一列
        X=temp
        #bias初值
        y=np.array(y) #列表转数组

        for k in range(epochs):
            #随机抽取每行X.shape[0]表示有多少行
            i=np.random.randint(X.shape[0])
            a=[X[i]] #取i行赋值给a
            #更新的实例
            #正向更新
            for l in range(len(self.weights)):
                a.append(self.activation(np.dot(a[l],self.weights[l]))) #dot是内积，对应相乘再相加
            error=y[i]-a[-1]#反向传送最后一个错误率 (Tj-Oj)
            deltas=[error*self.activation_deriv(a[-1])]
            #输出层Errj=Oj(1-Oj)(Tj-Oj)
            #根据误差反向传送
            #隐藏层
            for l in range(len(a)-2,0,-1):
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))
            deltas.reverse() #颠倒顺序
            #更新权重
            for i in range(len(self.weights)):
                layer=np.atleast_2d(a[i])
                delta=np.atleast_2d(deltas[i])
                self.weights[i]+=learning_rate*layer.T.dot(delta)

    def predict(self,x):
        x=np.array(x)
        temp=np.ones(x.shape[0]+1)
        temp[0:-1]=x
        a=temp
        for l in range(0,len(self.weights)):
            a=self.activation(np.dot(a,self.weights[l]))
        return a
