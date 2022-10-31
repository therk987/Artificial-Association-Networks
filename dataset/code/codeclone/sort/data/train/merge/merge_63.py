def merge(a):
    if len(a)>1:
        mid=len(a)//2
        left = a[:mid]
        right = a[mid:]
        merge(left)
        merge(right)
        i,j,k = 0,0,0
        while (k<len(a) and (i<=len(left)-1 and j<=len(right)-1)):
            if(left[i]<right[j]):
                a[k]=left[i]
                i+=1
            else:
                a[k]=right[j]
                j+=1
            k+=1
        
        if i == len(left)-1:
            for i in range(i,len(left)):
                a[k]=left[i]
                k+=1
        else:
            for j in range(j, len(right)):
                a[k]=right[j]
                k+=1
    
    return a
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(merge(INPUT_VALUE))