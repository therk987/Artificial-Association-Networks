# -*- coding: utf-8 -*-


#classic merge sort!!

def merge(left,right):
    sorted=[] #array to store the sorted list
    i,j=0,0
    while i<len(left) and j<len(right):
        if left[i] < right[j]:
            sorted.append(left[i])
            i+=1
        else:
            sorted.append(right[j])
            j+=1

    sorted+=left[i:]
    sorted+=right[j:]
    return sorted

def merge_sort(li):
    "function to compute merge-sort"
    if len(li)==1:
        return li
    middle=len(li)//2
    left_li=merge_sort(li[:middle])
    right_li=merge_sort(li[middle:])
    return merge(left_li,right_li)

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(merge_sort(INPUT_VALUE))