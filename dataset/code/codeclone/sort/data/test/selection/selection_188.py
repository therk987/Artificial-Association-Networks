def selectionSort(vetor):
    for ultimaPos in range(len(vetor)-1,0,-1):
        posMaior=0
        for pos in range(1,ultimaPos+1):
            if vetor[pos]>vetor[posMaior]:
                posMaior = pos

        aux = vetor[ultimaPos]
        vetor[ultimaPos] = vetor[posMaior]
        vetor[posMaior] = aux
    return vetor

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(selectionSort(INPUT_VALUE))