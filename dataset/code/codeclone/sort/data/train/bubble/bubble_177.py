import time
def bubblesort(slist):
    stime=time.time()
    swap = 1
    sortlist=slist
    while swap!=0:
        swap=0
        for i in range(len(sortlist)-1):
            if sortlist[i]>sortlist[i+1]:
                sortlist[i],sortlist[i+1]=sortlist[i+1],sortlist[i]
                swap=1
    print("sorted in:",time.time()-stime,"seconds" )
    return sortlist

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(bubblesort(INPUT_VALUE))