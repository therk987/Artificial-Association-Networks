"""
by Felipe Fernandes - Computer Engineer
Ordenação de list numérica com o Algoritmo Selection Sort

Sorting an array list with Selection Sort Algorithm
"""

# A simple for loop which iterates over the input answers and append the random values into the array

# Sorting Function
def selection_sort_algorithm(list):
    # Starting the timer of the code.
    # 'i' will iterate over all itens in the array
    for i in range(len(list)):
        # variable 'lowest_value' will be our lowest value on the array, receiving 'i'
        lowest_value = i
        # we need to compare if 'j' is lower than 'i', looking over the array again, skipping the last iterated item which is 'i+1'
        for j in range(i+1, len(list)):
            # if 'j' is lower than 'lowest_value', 'j' will be our lowest value ('lowest_value') on the sorted array
            if list[j] < list[lowest_value]:
                lowest_value = j
        # In this case, our array is unordered. But if the array was previously ordered, we don't need to use this if statement.
        if (list[i] != list[lowest_value]):
            aux = list[i]
            list[i] = list[lowest_value]
            list[lowest_value] = aux
    # Ending the timer
    # Calculating the processing time spent in seconds
    # Printing the sorted array
    return list
# Finally, calling the function for sorting the list

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(selection_sort_algorithm(INPUT_VALUE))