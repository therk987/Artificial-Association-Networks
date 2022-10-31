import datetime

before = datetime.datetime.now()

def bubble_sort(arr, n):
    if (n <= 1):
        return arr
    else:
        for i in range(0, n-1):
            if(arr[i] > arr[i + 1]):
                arr[i], arr[i+1] = arr[i+1], arr[i]
        return bubble_sort(arr, n-1)


INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(bubble_sort(INPUT_VALUE,len(INPUT_VALUE)))