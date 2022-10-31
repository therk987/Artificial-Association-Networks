
def sort_collection(collection):
    less = []
    equal = []
    greater = []
    if len(collection) > 1:
        pivot = collection[0]
        for x in collection:
            if x < pivot:
                less.append(x)
            if x == pivot:
                equal.append(x)
            if x > pivot:
                greater.append(x)
        return(sort_collection(less) + equal + sort_collection(greater))
    else:
        return collection
    return collection


INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(sort_collection(INPUT_VALUE))