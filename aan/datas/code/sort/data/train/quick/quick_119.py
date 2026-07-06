#Partiton Code Begin
#Last element of the array is set as pivot
# i is set as the index of the smaller element in the array given
def Create_Partition(Input_Array,Start_Index,End_Index):
	i=(Start_Index-1)
	pivot=Input_Array[End_Index]

#Start_Index denotes the starting index picked
#End_Index denotes the ending index picked

	for j in range(Start_Index,End_Index):

		if Input_Array[j]<=pivot:
			i=i+1
			Input_Array[i],Input_Array[j]=Input_Array[j],Input_Array[i]
	Input_Array[i+1],Input_Array[End_Index]=Input_Array[End_Index],Input_Array[i+1]
	return(i+1)

#Recursive quick sort function
#Start Index gets low value value of input
#End index gets high value of input
def Quick_Sort_Algo(Input_Array,Start_Index,End_Index):
	if Start_Index<End_Index:
		pi=Create_Partition(Input_Array,Start_Index,End_Index)

		Quick_Sort_Algo(Input_Array,Start_Index,pi-1)
		Quick_Sort_Algo(Input_Array,pi+1,End_Index)
	return Input_Array
#Quick sort code complete

def Quick_Sort(list):
    return Quick_Sort_Algo(list,0,len(list)-1)


INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(Quick_Sort(INPUT_VALUE))