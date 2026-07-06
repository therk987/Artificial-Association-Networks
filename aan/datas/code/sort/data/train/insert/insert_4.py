
def insertion_sort(some_list):
    count = 0
    for x in range(1, len(some_list), 1):
        for j in range(x, 0, -1):
            if some_list[x-j] > some_list[x]:
                count += 1
                some_list[x], some_list[x-j] = some_list[x-j], some_list[x]
    print(count)
    print(some_list)
    
    return some_list
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(insertion_sort(INPUT_VALUE))