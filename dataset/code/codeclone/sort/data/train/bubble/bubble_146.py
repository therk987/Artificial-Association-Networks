import random
from datetime import datetime

def bubbleSort(unsorted):
	endCheck = 1

    #FOR TIMING
    # var d = new Date();
    # var startTime = d.getTime();
    #END TIMING

	for i in range(0,len(unsorted)):
		for n in range(0,(len(unsorted) - endCheck)):
			if unsorted[n] > unsorted[n+1]:
				temp = unsorted[n]
				unsorted[n] = unsorted[n+1]
				unsorted[n+1] = temp
		endCheck+=1
    
    # FOR TIMING
    # var fdate = new Date();
    # console.log("BUBBLE SORT: " + (fdate.getTime() - startTime) + "ms");
    # END TIMING
	return unsorted



INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(bubbleSort(INPUT_VALUE))