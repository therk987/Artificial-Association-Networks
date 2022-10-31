
def selectSort(lst):
	srted = 0
	while srted<len(lst)/2:
		small = srted
		large = srted
		for i in range(srted+1,len(lst)-srted):
			if lst[i]<lst[small]:
				small = i
			if lst[i]>lst[large]:
				large = i
		lst[srted], lst[small] = lst[small], lst[srted]
		lst[len(lst)-1-srted], lst[large] = lst[large],lst[len(lst)-1-srted]
		srted+=1
	return  lst

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(selectSort(INPUT_VALUE))