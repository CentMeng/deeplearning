import math

# 计算两点距离sqrt是开平方，pow是z值n次方，此处n值是(x1-x2)，n是2
def ComputeEuclideanDistance(x1,y1,x2,y2):
    d = math.sqrt(math.pow((x1-x2),2)+math.pow((y1-y2),2))
    return d

d_ag = ComputeEuclideanDistance(3,104,18,90)

print(d_ag)