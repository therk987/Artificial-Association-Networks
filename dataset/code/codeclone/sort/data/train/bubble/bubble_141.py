# Bubble Sort


def bubbleSort(numbers):
    # DON'T EVER USE THIS

    print("Start:\n"+str(numbers)) # Print list at start
    temp = 0
    sL = "false"

    # Algorithm
    while sL != "true":
        sL = "true"
        for x in range(0, len(numbers)-1):
            if numbers[x] > numbers[x+1]:
                sL = "false"
                temp = numbers[x]
                numbers[x] = numbers[x+1]
                numbers[x+1] = temp
    # Output            
    print ("End:\n"+str(numbers)) # Print list at end
    return numbers



INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(bubbleSort(INPUT_VALUE))