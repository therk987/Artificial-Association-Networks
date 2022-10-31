def merge_sort(marks):
    top=len(marks)
    return merge_sort_r(marks, 0, top-1)

def merge (marks, first, last, sred):
    helper_list =[]

    i=first
    j=sred+1
    while i<= sred and j<=last:
        if marks[i]<= marks[j]:
            helper_list.append(marks[i])
            i=i+1
        else:
            helper_list.append(marks[j])
            j=j+1
    while i <=sred:
        helper_list.append(marks[i])
        i=i+1
    while j<= last:
        helper_list.append(marks[j])
        j=j+1
    for k in range(0, last-first+1):
        marks[first+k]=helper_list[k]
    


def merge_sort_r(marks, first, last):
    if first<last:
        sred=((first+last)//2)
        merge_sort_r(marks, first,sred)
        merge_sort_r(marks, sred+1,last)
        merge(marks,first,last,sred)
    return marks

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(merge_sort(INPUT_VALUE))