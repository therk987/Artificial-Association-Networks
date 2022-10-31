
def shellSort(vector): 
    n = len(vector) 
    orign = n//2
    while orign > 0: 
  
        for i in range(orign,n): 
            temp = vector[i] 
            j = i 
            while  j >= orign and vector[j-orign] >temp: 
                vector[j] = vector[j-orign] 
                j -= orign 
  
            vector[j] = temp 
        orign //= 2
    return vector

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(shellSort(INPUT_VALUE))