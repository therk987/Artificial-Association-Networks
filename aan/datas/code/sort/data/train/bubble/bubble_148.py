# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 14:14:59 2013

@author: David Moodie
"""


def bubble(numbers): # Bubble sort implementation
    """Sort list using bubble sort."""
    nums = len(numbers)
    for i in range(nums):
        for j in range( i + 1, nums):
            if numbers[j] < numbers[i]: #is the next num less than the current
                numbers[j], numbers[i] = numbers[i], numbers[j] #if yes, swap
    return numbers
                


INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(bubble(INPUT_VALUE))