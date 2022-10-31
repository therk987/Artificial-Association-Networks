from random import randrange
import datetime


def partition(arr):
	l = []; r= []; m = [];
	for i in range(len(arr)):
		if(arr[i]>arr[0]):
			r.append(arr[i]);
		elif(arr[i]<arr[0]):
			l.append(arr[i]);
		elif(arr[i] == arr[0]):
			m.append(arr[i]);
	return (l,m,r);

def quickSort(arr):
	if(len(arr) ==0):
		return [];
	if(len(arr) ==1):
		return arr;
	else:
		(l,m,r) = partition(arr);
		return quickSort(l)+ m + quickSort(r);



INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(quickSort(INPUT_VALUE))