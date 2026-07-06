# InsertionSort, O(n2)

#insertion sort list 
def insertion(A):
	for i in range(1, len(A)):  
		insert (A, i, A[i])
	return A

#insert value into proper location 
def insert (A, idx, value): 
	i = idx - 1 
	while i >= 0 and A[i] > value:
		A[i+1] = A[i]
		i = i - 1
	A[i+1] = value



INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(insertion(INPUT_VALUE))