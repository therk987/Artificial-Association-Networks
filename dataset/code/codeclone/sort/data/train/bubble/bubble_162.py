#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 22:36:10 2019

@author: cannybiscuit
"""

def bubble_sort_algorithm(array):
    
    n = len(array)
    
    # Loop through the entire array
    for i in range(n):

       # Here we repeatedly step through the list comparing adjacent pairs
         # If an element is larger than the element to the right of it will get swapped
         # The array is traversed until all elements have been sorted
         
        for j in range(0, n-i-1):
        
         # EXAMPLE:
         # arr = [1, 5, 2, 6, 4]
         # first iteration => [1,2,5,4,6]
         # Second iteration = [1,2,4,5,6]
            
            if array[j] > array[j+1]:
                array[j], array[j+1] = array[j+1], array[j]
                

    return array
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(bubble_sort_algorithm(INPUT_VALUE))