def Exchange(list, index1, index2):
	temp = list[index1]
	list[index1] = list[index2]
	list[index2] = temp

def Median(list, index1, index2, index3):
	if list[index1] > list[index2]:
		if list[index3] > list[index1]:
			return index1
		elif list[index3] > list[index2]:
			return index3
		else:
			return index2
	else:		#list[index1] <= list[index2]
		if list[index3] > list[index2]:
			return index2
		elif list[index3] > list[index1]:
			return index3
		else:
			return index1
		
def QuickSort(list, start, end):
	#Recursion end
	if start >= end:
		return

	#Choose the median from list[start], list[(start + end) / 2] and list[end] to improve QuickSort
	mid = Median(list, start, (start + end) // 2, end)
	Exchange(list, mid, end)

	#Divide list into two part
	m = start
	for i in range(start, end):
		if list[i] < list[end]:
			Exchange(list, m, i)
			m = m + 1
	Exchange(list, m, end)

	#Recursion
	QuickSort(list, start, m - 1)
	QuickSort(list, m + 1, end)
	return list
    
def Quick(list):
	return QuickSort(list,0,len(list)-1)

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(Quick(INPUT_VALUE))