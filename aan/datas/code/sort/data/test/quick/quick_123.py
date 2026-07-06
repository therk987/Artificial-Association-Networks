def quick_sort(list, start, end ):
    if start < end:
        index = sort(list,start,end)
        quick_sort(list,start,index)
        quick_sort(list,index+1,end)

    return list

def sort(list,start,end):
    pivot = list[start]        #?????????
    while start < end:         #????????????
        while start < end and list[end] >= pivot:       #????
            end -= 1
        while start < end and list[end] < pivot:        #????
            list[start] = list[end]
            start += 1
            list[end] = list[start]                 #?????==????
        list[start] = pivot                         #??????????
    return start



INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(quick_sort(INPUT_VALUE,0,len(INPUT_VALUE)-1))