
def BubbleSort(list):
    for iteration in range(len(list) - 1): 
        for i in range(len(list) - 1): 
            if list[i] > list[i + 1]:
                temp = list[i] 
                list[i] = list[i +1] 
                list[i + 1] = temp 
    return list
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(BubbleSort(INPUT_VALUE))