
def merge(a, b):
    """ Merge two arrays """

    c = []
    while a and b:
       
        if a[0] < b[0]:
            c.append(a[0])
            a.remove(a[0])
        else:
            c.append(b[0])
            b.remove(b[0])
   
    if not a:
        c += b
    else:
        c += a
    return c

def merge_sort(data):
    """ Sort @data using the Merge Sort technique """

    if not data or len(data) == 1:
       
        return data
    else:
        
        middle = int(len(data) / 2)
        a = merge_sort(data[:middle])
        b = merge_sort(data[middle:])
    
        return merge(a, b)

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(merge_sort(INPUT_VALUE))