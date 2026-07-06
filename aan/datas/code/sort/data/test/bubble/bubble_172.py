# -*- coding:utf-8 -*-
#简单的对python中的数据进行学习。
#先写一个冒泡算法
def bubbleSort(a):
    for waiceng in range(len(a)):
        for neiceng in range(len(a)-waiceng-1):
            if a[neiceng] >a[neiceng+1]:
                val=a[neiceng]
                a[neiceng] = a[neiceng+1]
                a[neiceng+1] =val
    return a

    

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(bubbleSort(INPUT_VALUE))




