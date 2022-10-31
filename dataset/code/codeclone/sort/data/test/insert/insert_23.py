
def inserctionSort(lista):
	for i in range(1,len(lista)):
		valor=lista[i]
		j = i
		while j>0 and lista[j-1]>valor:
			lista[j]=lista[j-1]
			j=j-1
		lista[j]=valor
	return lista

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(inserctionSort(INPUT_VALUE))