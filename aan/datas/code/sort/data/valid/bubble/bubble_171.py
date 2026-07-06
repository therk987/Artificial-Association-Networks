import random

def bubbleSort(lis):
	counter = 0
	maxRange = len(lis)-1
	for i in range(maxRange):
		if lis[i] > lis[i+1]:
			temp = lis[i+1]
			lis[i+1] = lis[i]
			lis[i] = temp
			counter += 1
	while counter>0:
		maxRange -= 1
		counter = 0
		for i in range(maxRange):
			if lis[i] > lis[i+1]:
				temp = lis[i+1]
				lis[i+1] = lis[i]
				lis[i] = temp
				counter += 1
	return lis



INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(bubbleSort(INPUT_VALUE))