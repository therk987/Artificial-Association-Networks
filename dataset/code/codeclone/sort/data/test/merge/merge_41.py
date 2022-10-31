def sort(l: list) -> list:
    if len(l) <= 1:
        return l

    a, b = sort(l[0::2]), sort(l[1::2])

    return __merge_lists(a, b)

def __merge_lists(a: list, b: list) -> list:

    result = list()
    i = j = k = 0

    len_a, len_b = len(a), len(b)

    while k < (len_a + len_b):

        if a[i] < b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1

        if i == len_a:
            result += b[j:]
            break

        if j == len_b:
            result += a[i:]
            break

        k += 1

    return result


INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(sort(INPUT_VALUE))