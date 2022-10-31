#=============================================
# A merge sorter for comparable elements
# - by Alexander Olson.

#Mergesort is a simple and very common algorithm for sorting comparable elements.

#The algorithm has quite a few variations: you can read more about it in CLRS (or by its proper title, "Introduction to Algorithms" by
#Charles E. Leiserson, Clifford Stein, Ronald Rivest, and Thomas H. Cormen.

#Here, we implement it in python: we'll assume that we already have a list of elements, and- if they're not already integers-
#a function to cheaply compare them.

#If you'd like to use this for whatever (non-plagiarizing) reason, feel free! (But please attribute it).
#=============================================

#---------------------------------------------
#USAGE
#---------------------------------------------

#radixsort(A, f) will take two inputs:

# 1) A : the *list* of n inputs to be sorted.

# 2) f (optional) : a *function* that compares any two elements of the list, and decides if x <= y (returns yes/no.)
# If you don't provide f, we'll use the standard comparison x <= y instead.

# Note that your function f will have time complexity f(n): mergesort performs O(n lg(n)) comparisons, for O(f(n) * n lg(n))
# time in total.

# WARNING: If f doesn't return a boolean when comparing two elements, you'll almost certainly run into errors!

# Time complexity is O(f(n) * n lg(n)), or just O(n lg(n)) if f(n) is O(1).

#Once radixsort() is done, it'll return the elements sorted by the numbering given by f.

#---------------------------------------------
#CODE
#---------------------------------------------

def mergesort(A, f=None):
	#If not given an f, use the identity instead.
	if f==None:
		def f(x, y): 
			return x <= y
			
	n = len(A)
	merge(A, f, 0, n)
	return A	

def merge(A, f, l, r):
	if r - l > 1:
		#Merge the left and right, and combine the two sorted lists.
		m = l + ((r-l)//2)
		merge(A, f, l, m)
		merge(A, f, m, r)
		B = []
		j = l
		k = m
		for i in range(l, r):
			if k >= r or (j < m and f(A[j],A[k])):
				B.append(A[j])
				j = j + 1
			else:
				B.append(A[k])
				k = k + 1
		#Copy B over the original array.
		for i in range(l, r):
			A[i] = B[i-l]

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(mergesort(INPUT_VALUE))