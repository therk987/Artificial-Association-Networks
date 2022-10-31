def merge(a,b):
	result = []
	i = 0
	j = 0
	while i<len(a) and j<len(b):
		if a[i]<b[j]:
			result.append(a[i])
			i+=1
		else:
			result.append(b[j])
			j+=1
	if i==len(a):
		result+=b[j:]
	else:
		result+=a[i:]

	return result

def merge_sort(d):
	len_d = len(d)
	if len_d < 2:
		return d
	mid = len_d // 2	
	print(d[:mid],d[mid:])
	a1 = merge_sort(d[:mid])
	a2 = merge_sort(d[mid:])
	result=merge(a1,a2)
	return result	

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(merge_sort(INPUT_VALUE))