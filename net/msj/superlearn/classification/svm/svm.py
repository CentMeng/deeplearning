from sklearn import svm

X = [[2,0],[1,1],[2,3]]
Y = [0,0,1] #区分类别，如图假如已知[2,0]和[1,1]点都是同一类，即0类，[2,3]是一类即1类
clf = svm.SVC(kernel='linear')
clf.fit(X,Y) #建立模型，已经建立了超平面

print(clf)

print(clf.support_vectors_) #支持向量[[ 1.  1.] [ 2.  3.]]

print(clf.support_) #支持向量点在X中的位置，第一个和第二个 [1 2]

print(clf.n_support_) #每个类中有多少个点是支持向量 [1 1]

print(clf.predict([0,1]))