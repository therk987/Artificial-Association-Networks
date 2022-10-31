
def merge_sort(lista):
    if len(lista) <= 1:
        return lista
    mitad = len(lista) // 2
    izquierda = merge_sort(lista[:mitad])
    derecha = merge_sort(lista[mitad:])
    return merge(izquierda, derecha)

def merge(izquierda, derecha):
    if not izquierda:
        return derecha
    if not derecha:
        return izquierda
    if izquierda[0] < derecha[0]:
        return [izquierda[0]] + merge(izquierda[1:], derecha)
    return [derecha[0]] + merge(izquierda, derecha[1:])



INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(merge_sort(INPUT_VALUE))