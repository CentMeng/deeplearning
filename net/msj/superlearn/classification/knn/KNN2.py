# 导入文件，使用自己数据集
## [-1] 表示最后一列
## ()代表tuple元祖数据类型，元祖是一种不可变序列
## []代表list列表数据类型，列表是一种可变序列
## {}代表dict字典数据类型，字典是Python中唯一内建的映射类型。字典中的值没有特殊的顺序，但都是存储在一个特定的键（key）下 类似于Map
## iterm()将一个字典以列表的形式返回，因为字典是无序的，所以返回的列表也是无序的 3.x之后iterm代替iteritems（）
### a = {'a':1,'b':3}
### a.items()
### 返回a = [('a',1),('b',3)]
## iteritems()返回一个迭代器
### b = a.iteritems()
### list(b) =[('a',1),('b',3)]
### for k,v in b:
###     print k,v
###  返回a 1
###     b 3
import csv
import random
import math
import operator

# 装载数据，文件路径名，训练集和测试集分隔行数，每一行的列数，训练集，测试集
def loadDataSet(fileName,split,columns,trainingSet=[],testSet=[]):
    with open(fileName,'rt') as file:
        lines = csv.reader(file)
        dataSet = list(lines)
        for row in range(len(dataSet)-1):
            for column in range(columns):
                dataSet[row][column] = float(dataSet[row][column])
            if random.random()<split:
                trainingSet.append(dataSet[row])
            else:
                testSet.append(dataSet[row])

# 计算路径
def euclideanDistance(instance1,instance2,length):
    distance = 0
    for i in range(length):
        distance += pow((instance1[i]-instance2[i]),2)
    return math.sqrt(distance)

# 取k个结果
def getNeighbors(trainingSet,testInstance,k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance,trainingSet[x],length)
        distances.append((trainingSet[x],dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

# 获取结果
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1] #-1是指最后一个值
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sorteVotes = sorted(classVotes.items(),key = operator.itemgetter(1),reverse=True)
    return sorteVotes[0][0]

# 训练集正确率
def getAccuary(testSet,predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct +=1
    return (correct/float(len(testSet)))*100.00

def main():
    trainingSet = []
    testSet = []
    # 2/3数据为训练集，1/3数据为测试集
    split = 0.67
    loadDataSet(r'/users/vincent/Documents/project/python/deeplearning/net/msj/superlearn/classification/knn/resources/iris.csv',split,4,trainingSet,testSet) #加r是忽略字符串里面特殊符号
    print('TranSet: ',repr(len(trainingSet)))
    print('TestSet: ',repr(len(testSet)))

    predictions = []
    k = 3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet,testSet[x],k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('> predicted= '+ repr(result) +', actual = ' + repr(testSet[x][-1]))
    accuary = getAccuary(testSet,predictions)
    print('Accuary: '+ repr(accuary) + '%')

main()