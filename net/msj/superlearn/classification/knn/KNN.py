# 使用sklearn自有数据集
from sklearn import neighbors
from sklearn import datasets

knn = neighbors.KNeighborsClassifier()

iris = datasets.load_iris()

# 特征变量
print("data:",iris.data)

# 特征分类结果值
print("target: ",iris.target)

# 建立模型
knn.fit(iris.data,iris.target)

predictedLabel = knn.predict([[0.1,0.2,0.3,0.4]])

print(predictedLabel)