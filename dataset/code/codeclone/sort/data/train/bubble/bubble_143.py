def bubble_sort(lista):
    n = len(lista)
    while True:
        trocado = False
        for i in range(1, n):
            if lista[i-1] > lista[i]:
                lista[i-1], lista[i] = lista[i], lista[i-1]
                trocado = True
        n -= 1
        if not trocado:
            break
    return lista
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(bubble_sort(INPUT_VALUE))