
def selectionSort(arr):
	for i in range(0,len(arr)):
		lowest = arr[i]
		index = i
		for n in range(1+i, len(arr)):
			if arr[n] < lowest:
				#perform swap
				index = n
				lowest = arr[n]
		temp = arr[i]
		arr[i] = lowest
		arr[index] = temp
	arr[index] = temp
	return arr



INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(selectionSort(INPUT_VALUE))