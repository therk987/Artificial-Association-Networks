def sort(list):
    # type: (object) -> object
    for index in range(1, len(list)):
        value = list[index]
        i = index - 1
        while i >= 0 and (value < list[i]):
            list[i + 1] = list[i]
            list[i] = value
            i = i - 1

    return list
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(sort(INPUT_VALUE))