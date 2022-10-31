def bubblesort(data):

    """
    Takes a list of numbers and runs the bubblesort
    algorithm on them, printing each stage to show progress.
    """

    print("Unsorted...")

    print_data(data, len(data))

    print("Bubble sorting...")

    lastindex = len(data) - 1

    while lastindex > 0:

        for i in range(0, lastindex):

             if data[i] > data[i+1]:

                data[i] = data[i] ^ data[i+1]
                data[i+1] = data[i] ^ data[i+1]
                data[i] = data[i] ^ data[i+1]

        print_data(data, lastindex)

        lastindex-= 1

    print("Sorted!")
    return data


def print_data(data, sortedto):

    """
    Prints data on a single line.
    Takes a sortedto argument, printing the sortedto
    portion in green and the unsorted portion in red.
    """

    for i in range(0, len(data)):

        if i >= sortedto:
            print("\x1B[32m{:3d}\x1B[0m".format(data[i]), end="")
        else:
            print("\x1B[31m{:3d}\x1B[0m".format(data[i]), end="")

    print("")


INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(bubblesort(INPUT_VALUE))