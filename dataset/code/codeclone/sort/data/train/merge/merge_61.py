#code to be used to merge the two sides of inputs from mergesort()
def merge(left, right):
    result = []
    i, j = 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            print((result), "left")
            i += 1
        else:
            result.append(right[j])
            print((result),"right")
            j += 1

    result += left[i:]
    result += right[j:]
    print(result, "result")
    return result


def mergesort(lst):
    counter = 0
    if (len(lst) <= 1):
        counter += 1
        return lst

    mid = int(len(lst) / 2)
    leftie = mergesort(lst[:mid])
    rightie = mergesort(lst[mid:])
    print((counter) ,leftie,rightie    )
    return merge(leftie, rightie)



INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(mergesort(INPUT_VALUE))