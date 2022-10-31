
def selectionsort(arr):
    minimum = arr[0]
    min_index = 0
    #print arr
    for j in range(0, len(arr)):
        minimum = arr[j]
        min_index = j
        for i in range(j, len(arr)):
            if arr[i] < minimum :
                minimum = arr[i]
                min_index = i
        (arr[j],arr[min_index]) = (arr[min_index],arr[j])
        #print (str(arr)),
        #print ("; swapped indicies :"+str(j)+" ,"+str(min_index)+"; swapped values :"+str(arr[j])+", "+str(arr[min_index]))
   
    return arr


INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(selectionsort(INPUT_VALUE))