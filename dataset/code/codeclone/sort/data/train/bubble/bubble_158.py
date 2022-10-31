"""
by Felipe Fernandes - Computer Engineer
Ordenação de list numérica com o Algoritmo Bubble Sort
Sorting an array list with Bubble Sort Algorithm
"""

# Sorting Function
def bubble_sort_algorithm(list):
    LenghtOfTheArray = len(list)

    # Iterate through all array elements
    for i in range(LenghtOfTheArray):

        # Last 'i' elements are in their correct places
        for j in range(0, LenghtOfTheArray-i-1):
            
            # Iterate the array from the index 0 to 'LenghtOfTheArray'-i-1
            # Swap only if the found element is greater than the next
            if list[j] > list[j+1]:
                list[j], list[j+1] = list[j+1], list[j]
    print(list)
    return list


INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(bubble_sort_algorithm(INPUT_VALUE))