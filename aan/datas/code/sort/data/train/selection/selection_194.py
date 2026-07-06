
def selectionsort(some_list):
    count = 0
    for x in range(len(some_list)):
        minimum = some_list[x]
        for j in range(x, len(some_list), 1):
            if some_list[j] < minimum:
                count += 1
                minimum = some_list[j]
                some_list[j], some_list[x] = some_list[x], some_list[j]
        print(x)
    print(count)
    return(some_list)

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(selectionsort(INPUT_VALUE))