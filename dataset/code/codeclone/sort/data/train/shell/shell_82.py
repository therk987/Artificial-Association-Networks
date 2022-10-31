
def encontraSeparacoes (v):
    sep = []
    atual = 1
    sep.append(atual)
    while atual < len(v):
        atual = (3 * atual) + 1
        sep.append(atual)
    sep.pop()
    sep.reverse()

    return sep


def shellSort (v):
    separacoes = encontraSeparacoes(v)
    for separacao in separacoes:
        for i in range(separacao, len(v), 1):
            j = i - separacao
            if v[i] < v[i - separacao]:
                while j >= 0 and v[j] > v[i]:
                    j -= separacao
                v.insert(j + separacao, v.pop(i))

    return v

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(shellSort(INPUT_VALUE))