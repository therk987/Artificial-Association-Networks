def merge(left, right):
    global k
    if not len(left) or not len(right):
        return left or right
    #if len(left)>1 and len(right)>1:
    result = []
    i, j = 0, 0
    while (len(result) < len(left) + len(right)):
        if left[i] < right[j]:
            result.append(left[i])
            i+= 1
        else:
            result.append(right[j])
            j+= 1
        if i == len(left) or j == len(right):
            result.extend(left[i:] or right[j:])
            break    
    return result

def mergesort(list):
    if len(list) < 2:
        return list
    middle = len(list)//2
    left = mergesort(list[:middle])
    right = mergesort(list[middle:])

    return merge(left, right)
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(mergesort(INPUT_VALUE))