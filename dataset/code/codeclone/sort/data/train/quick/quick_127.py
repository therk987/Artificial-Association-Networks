#! /usr/bin/env python
# -*- coding: utf-8 -*-



def quickSort(vector):
    return sort(vector,0,len(vector)-1)

def sort(vector,x,y):
    if x<y:

        punto = particion(vector,x,y)

        sort(vector,x,punto-1)
        sort(vector,punto+1,y)
    return vector

def particion(vector,x,y):
    pivote = vector[x]

    izquierda = x+1
    derecha = y

    done = False
    while not done:

        while izquierda <= derecha and vector[izquierda] <= pivote:
             izquierda = izquierda + 1

        while vector[derecha] >= pivote and derecha >= izquierda:
            derecha = derecha -1

        if derecha < izquierda:
            done = True
        else:
            aux = vector[izquierda]
            vector[izquierda] = vector[derecha]
            vector[derecha] = aux

    aux = vector[x]
    vector[x] = vector[derecha]
    vector[derecha] = aux


    return derecha
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(quickSort(INPUT_VALUE))