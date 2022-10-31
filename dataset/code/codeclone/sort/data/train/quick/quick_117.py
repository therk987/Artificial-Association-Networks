'''
Created on 12 de ago de 2016

@author: Hugo
'''

def qsort(a):
    if len(a)<=1:
        return a
    menor, igual, maior = [], [], []
    pivo = a[0]
    for i in a:
        if i < pivo:
            menor.append(i)
        elif i == pivo:
            igual.append(i)
        else:
            maior.append(i)
    return qsort(menor) + igual + qsort(maior)
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(qsort(INPUT_VALUE))