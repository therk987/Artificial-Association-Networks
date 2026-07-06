
def insertionSort(A):
    print ("\nOrdering...")
    for j in range (1,len(A)):
        print ("-------- ")
        print ("\tkey= A[j]=",A[j])
        print ("\ti= j-1 =",str(j-1))
        key=A[j];
        i=j-1;
        while i>=0 and A[i]>key:
            A[i+1]=A[i]
            print ("\t", A, "##",)
            print ("i:",i,)
            print ("#",)
            print ("j:",j,)
            print ("#",)
            print ("key:",key)
            i-=1
        print ("\tExchange A[i+1]=",str(A[i+1]),"with",str(key))
        A[i+1]=key    
    print ("--------")
    return A

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(insertionSort(INPUT_VALUE))