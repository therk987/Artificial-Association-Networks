def selectionSort(arr):
    for i in range(len(arr)):
        min = i
        for b in range(i+1, len(arr)):
            if arr[b] < arr[min]:
                min = b
        arr[i] , arr[min] = arr[min], arr[i] 
  
    return arr
  

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(selectionSort(INPUT_VALUE))