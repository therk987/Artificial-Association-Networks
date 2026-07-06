def splitbyn(listvar, parts):
	newlist = []
	midpoint = int(len(listvar)/parts)
	if int(len(listvar)%parts) >= int(parts/2):
		midpoint += 1
	pos = 0
	for i in range(1,parts):
		nextsl = pos+midpoint
		newlist.append(listvar[pos:nextsl])
		pos = nextsl
	newlist.append(listvar[pos:])
	return newlist


def MergeSort(src_array, splitby=2, debug=False):
	sorted = []
	if len(src_array) < 2: 
		return src_array
	elif splitby < 2:
		print("Cannot split into less than 2 parts")
		return src_array
	for subarray in splitbyn(src_array, splitby):
		if debug:
			print("Split into "+ str(subarray))
		sorted = Merge(sorted,MergeSort(subarray,splitby,debug),debug)
	
	return sorted

def Merge(llist, rlist,debug=False):
	retlist = []
	
	while(len(llist) > 0 and len(rlist) > 0):
		if llist[0] < rlist[0]:
			retlist.append(llist[0])
			if debug:
				print(str(llist[0])+' taken from left over '+str(rlist[0]))
			del llist[0]

		else:
			retlist.append(rlist[0])
			if debug:
				print(str(rlist[0])+' taken from right over '+str(llist[0]))
			del rlist[0]
				
	return retlist+llist+rlist
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(MergeSort(INPUT_VALUE))