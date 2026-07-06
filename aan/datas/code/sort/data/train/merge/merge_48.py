global licznik

def merge(X, Y):

    # type: (list, list) -> list
    Z = []
    i = 0
    j = 0
    inv = 0

    m = max(len(X), len(Y))-1
    for k in range(0, 2*(m+1)):
        if i <= m and j <= m:
            if X[i] < Y[j]:
                Z.extend([X[i]])
                i += 1
            else:  # brak zabezpieczenia dla takich samych wartosci
                Z.extend([Y[j]])
                j += 1
                #if i <= m:# Gdy i !=m tzn. jakas wartosc w Y jest mniejsza niz w X.
                inv +=1*(m+1-i)
        elif i > m:
            Z.extend([Y[j]])
            j += 1
        elif j >m:
            Z.extend([X[i]])
            i += 1


    return Z,inv

def merge_sort(A):
    V1 = []
    V2 = []
    inv_total = 0
    n = len(A)-1
    odd = False
    if len(A)%2 !=0 and n !=0:
        odd = True
        temp = min(A)
        if A[0]!= min(A):
            inv_total +=(A.index(min(A)))
        A.remove(temp)
        n-=1
    if n+1 == 1:
        return [A[0]],0
        #print ('a')
    else:
        B = A[0:n//2+1]
        C = A[n//2+1:n+1]
        B1,inv1 = merge_sort(B)
        V1.extend(B1)
        C2,inv2 = merge_sort(C)
        V2.extend(C2)
    Z,inv_mix=merge(V1,V2)
    inv_total += inv1+inv2+inv_mix
    if odd == True:
        Z.insert(0,temp)
        odd = False
    return Z,inv_total


INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(merge_sort(INPUT_VALUE))