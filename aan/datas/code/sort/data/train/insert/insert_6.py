# -*- coding: utf-8 -*-
#algoritmo de insertion sort


def insertionSort(arreglo):
	if len(arreglo) <= 1:
		return
	for x in range(1,len(arreglo)):
		key = arreglo[x]
		indice = x 
		while indice > 0 and key < arreglo[indice - 1] :
			arreglo[indice] = arreglo[indice-1]
			indice = indice-1
		arreglo[indice] = key
	return arreglo	
		


INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(insertionSort(INPUT_VALUE))