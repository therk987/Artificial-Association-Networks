import random

def bubblesort(random_list):
    count = 0
    while count < len(random_list):
        for x in range(0, len(random_list)-1):
            if random_list[x] > random_list[x+1]:
                random_list[x], random_list[x+1] = random_list[x+1], random_list[x]
        count += 1
    return random_list
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(bubblesort(INPUT_VALUE))