def merge(b,c,a):
	i,j,k = 0,0,0
	while i < len(b) and j < len(c):
		if b[i] <= c[j]:
			a[k] = b[i]
			i += 1
		else:
			a[k] = c[j]
			j += 1
		
		k += 1
	
	if i == len(b):
		a[k:] = c[j:]
	else:
		a[k:] = b[i:]
	return b
#Merge Sort Algorithm
#https://en.wikipedia.org/wiki/Merge_sort
def merge_sort(a):
	if len(a) > 1:
		b= a[:(len(a)//2)]
		c= a[(len(a)//2):]
		merge_sort(b)
		merge_sort(c)
		merge(b,c,a)

		return a

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(merge_sort(INPUT_VALUE))