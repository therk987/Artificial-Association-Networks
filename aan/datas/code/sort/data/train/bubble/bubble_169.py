import random
import datetime

def bubbleSort(list):
	for i in range(0, len(list)-1):
		for j in range(0, len(list)-i-1):
			if list[j] > list[j+1]:
				(list[j], list[j+1]) = (list[j+1], list[j])
	return list



INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(bubbleSort(INPUT_VALUE))