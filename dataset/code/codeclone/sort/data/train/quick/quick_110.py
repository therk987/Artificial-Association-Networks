
def partition(list_in, start, end):
    lower_than_idx = start - 1
    pivot = list_in[end]
    for j in range(start , end):
        if list_in[j] <= pivot:
            lower_than_idx += 1
            list_in[j], list_in[lower_than_idx] = list_in[lower_than_idx], list_in[j]

    list_in[end], list_in[lower_than_idx + 1] = list_in[lower_than_idx + 1], list_in[end]
    #print(pivot)
    #print(list_in.index(pivot))
    return lower_than_idx + 1


# python partition code from internet used for debugging
# def partition(arr, low, high):
#     i = (low - 1)  # index of smaller element
#     pivot = arr[high]  # pivot
#
#     for j in range(low, high):
#
#         # If current element is smaller than or
#         # equal to pivot
#         if arr[j] <= pivot:
#             # increment index of smaller element
#             i = i + 1
#             arr[i], arr[j] = arr[j], arr[i]
#
#     arr[i + 1], arr[high] = arr[high], arr[i + 1]
#     return i + 1


def quick_sort(list_in, start, end):

    if start < end:
        split = partition(list_in, start, end)
        # print(start, end, split)
        quick_sort(list_in, start, split - 1)
        quick_sort(list_in, split + 1, end)

    return list_in

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(quick_sort(INPUT_VALUE,start=0,end=len(INPUT_VALUE)-1))