
def sel_sort(a):
	b = [0 for i in range(len(a))]
	
	for i in range(0,len(a)):
		min = a[i]
		for j in range(i,len(a)):
			if (a[j] < min):
				tmp = min
				min = a[j]
				a[j] = tmp
			
			
		b[i] = min
		
	return b
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(sel_sort(INPUT_VALUE))