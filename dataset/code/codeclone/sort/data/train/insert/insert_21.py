import random
from datetime import datetime

def randomNumArrayGen(max, length):
	arr = []
	for count in range(length):
		arr.append(int(round(random.random()*max)))
	return arr


def insertionSort(arr):
	for i in range(len(arr)):
		tracker = 0
		for n in reversed(range(0, i)):
			print(arr[i], "<", arr[n])
			if arr[i] < arr[n]:
				print("swap!")
				tracker += 1
				print(arr)
		startPos = i
		for k in reversed(range(tracker)):

			(arr[startPos-1], arr[startPos]) = (arr[startPos],arr[startPos-1])
			startPos-=1
	return arr




INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(insertionSort(INPUT_VALUE))