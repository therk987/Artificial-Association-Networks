import copy

def bubble_sort(data_set):
    frames = [data_set]
    ds = copy.deepcopy(data_set)
    for i in range(len(data_set)-1):
        flag = False
        for j in range(len(data_set)-i-1):
            if ds[j] > ds[j+1]:
                ds[j], ds[j+1] =  ds[j+1], ds[j]
                flag = True
            frames.append(copy.deepcopy(ds))
        if not flag:
            break
    frames.append(ds)
    return ds
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(bubble_sort(INPUT_VALUE))