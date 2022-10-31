
def shellSort(tablica):
    """ shell sort"""
    N = len(tablica)
    h = int(N//2) #rozstaw


    while h >= 1:
        
        
        for i in range (0, N-h): # przejscie 'w przod' tablicy
            done = False
            j = i
            while(done == False):   #jezeli jest potrzeba przejscie 'w tyl' tablicy w celu znalezienia odpowiedniego miejsca
                
                if tablica[j] > tablica[j+h]:
                    #podmiana
                    tmp = tablica[j]
                    tablica[j] = tablica[j+h]
                    tablica[j+h] = tmp
                
                if j-h >= 0:    #przestawienie porownania o jeden rozstaw w tyl
                    j = j-h
                else:
                    done = True #znaleziono miejsce
                

        h = h = int(h//2) #zmniejszenie rozstawu


    return tablica






INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(shellSort(INPUT_VALUE))