# 导入数据
# 预处理数据（sklearn库使用决策树必须是采用01数据矩阵形式）
## 分离数据（分离出数据集，和结果数据集）
## 准备数据（数据集加标签为转义成矩阵数据做准备）
## 转义数据矩阵（使用DictVectorizer转义数据）
# 建模
# 生成图
# 预测

# 整型类型数据
from sklearn.feature_extraction import DictVectorizer
import csv # csv导入
from sklearn import preprocessing
from sklearn import tree # 决策树
from sklearn.externals.six import StringIO # io操作

#rb , rt解释
#r、w、a为打开文件的基本模式，对应着只读、只写、追加模式；
#b、t、+、U这四个字符，与以上的文件打开模式组合使用，二进制模式，文本模式，读写模式、通用换行符，根据实际情况组合使用

# 导入数据
allData = open(r'/users/vincent/Documents/project/python/deeplearning/net/msj/superlearn/classification/decisiontree/resources/decisiontree.csv','rt')
reader = csv.reader(allData)
headers = next(reader)
print(headers)


# 预处理数据开始 #
featureList = [] #装取不算最后一列（即结果）的原生值
labelList = []   #装取最后一列（即结果）的原生值

## 分离数据
for row in reader: #逗号隔开为一条数据
    #len(row)等同于row.size
    labelList.append(row[len(row)-1]) #labelList取每一行最后一个值

    rowDict = {} #把每一行的除最后一列数据存储为一个数组
    for i in range(1,len(row)-1):# 等同于for（int i=1;i<row.size;i++）
        # print(row[i])
        ## 准备数据
        rowDict[headers[i]] = row[i] #{'age': 'youth', 'income': 'high', 'student': 'no', 'credit_rating': 'fair'}
        # print("rowDict:",rowDict)
    featureList.append(rowDict)

print(labelList)
print(featureList)

# 转义数据，转化为0-1矩阵
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()

print("dummX:"+str(dummyX))
print(vec.get_feature_names())

print("labelList:"+str(labelList)) #输出标签集合

#把最后一列结果转化为0-1形式
lb = preprocessing.LabelBinarizer() #对于标称型数据来说，preprocessing.LabelBinarizer是一个很好用的工具。比如可以把yes和no转化为0和1，或是把incident和normal转化为0和1。
dummyY = lb.fit_transform(labelList)
print("dummyY: "+ str(dummyY))

# 预处理数据结束 #

#采用sklearn的ID3算法，建模
clf = tree.DecisionTreeClassifier(criterion='entropy') # 声明分离器，criterion='entropy'是选择采用哪种算法
clf = clf.fit(dummyX, dummyY) # 建模
print("clf：" + str(clf)) # 打印出分离器一些参数，可以在声明时候指定 tree.DecisionTreeClassifier(criterion='entropy',min_samples_leaf=2)

#打印tree
#输出tree的dot格式图，使用Graphviz转换成图像
with open("/users/vincent/Documents/project/python/deeplearning/net/msj/superlearn/classification/decisiontree/resources/allData.dot",'w') as f:
  f = tree.export_graphviz(clf,feature_names=vec.get_feature_names(), out_file = f) #feature_names=vec.get_feature_names()为了还原矩阵以前特征值


#dot -Tpdf allData.dot -o allData.pdf # Graphviz转成pdf命令

oneRowX = dummyX[0, :] # 取第一行数据做改动
#print("oneRow: " + str(oneRowX))

#预测
newRowX = oneRowX #将第一行数据age从0，0，1（年轻人）改成1 0 0（中年人）
newRowX[0] = 1
newRowX[1] = 0
newRowX[2] = 0
print("newRowX: " + str(newRowX))
predictedY = clf.predict(newRowX)
print("predicted: " + str(predictedY))


