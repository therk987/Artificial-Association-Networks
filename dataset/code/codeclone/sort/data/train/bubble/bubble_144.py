
def bubbleSort(numlist):
    for value in range(len(numlist)-1,0,-1):
        for i in range(value):
            if numlist[i]>numlist[i+1]:
                temp = numlist[i]
                numlist[i] = numlist[i+1]
                numlist[i+1] = temp
    return numlist

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(bubbleSort(INPUT_VALUE))



