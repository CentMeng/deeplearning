from numpy import genfromtxt
from sklearn import linear_model

dataPath = r"/Users/mengxiangcheng/Documents/project/python/deeplearning/net/msj/superlearn/linear/regression/resource/dilivery.csv"

deliveryDate = genfromtxt(dataPath,delimiter=',')

print('data')
print(deliveryDate)

X = deliveryDate[:,:-1] #:代表所有行， :-1从一开始到倒数第二列（-1是最后一列）
Y = deliveryDate[:,-1] #:代表所有行， -1最后一列

print("X")
print(X)
print("Y")
print(Y)

#创建线性回归模型
regr = linear_model.LinearRegression()

#建模
regr.fit(X,Y)

print("coefficients")
print(regr.coef_)
print("intercept:")
print(regr.intercept_) #截距

xPred = [102,6]
yPred = regr.predict(xPred)
print("predict y:")
print(yPred)


