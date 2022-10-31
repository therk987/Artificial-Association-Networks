def merge(lst):
    if len(lst) > 1:
        #mencari nilai tengah list
        mid = len(lst)//2
        #membagi list jadi 2 bagian 
        l = lst[0:mid]
        r = lst[mid:len(lst)]

        
        merge(l) #sort tengah pertama
        
        merge(r) #sort tengah kedua
        

        i = j = k = 0

        #menggabungkan data ke list setelah di-sort
        while i < len(l) and j < len(r):
            if l[i] < r[j]:
                lst[k] = l[i]
                i+=1
            else:
                lst[k] = r[j]
                j+=1
            k+=1
           
        
        #menggabungkan data yang masih tersisa
        while i < len(l):
            lst[k] = l[i]
            i+=1
            k+=1 

        while j < len(r):
            lst[k] = r[j]
            j+=1
            k+=1 
    return lst

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(merge(INPUT_VALUE))