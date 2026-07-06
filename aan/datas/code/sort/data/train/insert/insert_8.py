def insertionSort(List):
        for i in range(1, len(List)):
                currentNumber = List[i]
                for j in range(i-1, -1, -1):
                        if List[j] > currentNumber :
                                List[j], List[j+1] = List[j+1], List[j]
                        else:
                                List[j+1] = currentNumber
                                break
        return List

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(insertionSort(INPUT_VALUE))