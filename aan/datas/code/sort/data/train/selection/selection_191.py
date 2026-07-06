import random, string, timeit

'''
Determines the execution time of a selectionsort algorithm, given an 
array, length ARRAYLENGTH, of random strings, length STRINGLENGTH. 
All rigths and original code of the wrapper and wrapped functions 
belong to Xiaonuo Gantan from Python Central at 
"pythoncentral.io/time-a-python-function/". 
'''


def selectionsort(array):
	'''
	Selectionsort algorithm for sorting an array of strings
	Input: An array of strings, array
	Output: The sorted array of strings
	'''
	count = 0
	for i in range(len(array)):
		minPos = i
		for j in range(i, len(array)):
			count += 1
			if array[j] < array[minPos]:
				minPos = j
		temp = array[i]
		array[i] = array[minPos]
		array[minPos] = temp
	return array

#Assign variables

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(selectionsort(INPUT_VALUE))