def shellSort(nums):
    n = len(nums)
    h = 1
    
    while h<=n/3:
        h = h*3+1

    while h > 0:
            for i in range(h, n):
                c = nums[i]
                j = i
                while j >= h and c < nums[j - h]:
                    nums[j] = nums[j - h]
                    j = j - h
                    nums[j] = c
            h  = int(h / 2.2)
    return nums



# ALGUNS PONTOS QUE VALEM A PENA DAR ATENÇÃO
# linhas 5 e 6: define tamanho das partições baseado na sequência de intervalos de Knuth.
    #descobre o melhor h
# linha 8: o h representa o marco para o fim do intervalo
# linhas 12 a 15: Algoritmo Insertion Sort - Grupo 1
# linha 14: decrementa para percorrer outras partições
# linha 16: Isto é feito para diminuir o tamanho das partições até chegar a 0.

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(shellSort(INPUT_VALUE))