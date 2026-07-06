# 원래는 insertion 이었음 근데 내용물 보니 bubble 임ㅋㅋ


def swap(arr, i , j):
    temp = arr[j]
    arr[j] = arr[i]
    arr[i] = temp
    return arr

def bubble_sort(arr):
    i = 1
    temp = 0
    n = len(arr)
    while i < n:
        current = arr[i]
        j = i - 1
        while j >= 0:
            if current < arr[j]:
                arr = swap(arr,j,j+1)
                '''temp = arr[j]
                arr[j] = arr[i]
                arr[i] = temp'''
                print(arr)
            j = j -1
            current = arr[j+1]
        i = i + 1
    return arr

    
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(bubble_sort(INPUT_VALUE))