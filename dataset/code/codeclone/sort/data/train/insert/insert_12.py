import random
import datetime

def insertionSort(list):
	result = []
	for i in range(0, len(list)):
		for j in range(0, len(result)):
			if list[i] <= result[j]:
				result.insert(j, list[i])
				break
		else:
			result.append(list[i])
			

	return result





INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(insertionSort(INPUT_VALUE))