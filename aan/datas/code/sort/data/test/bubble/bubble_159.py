def swap(unsorted_list, x):
    temp = unsorted_list[x]
    unsorted_list[x] = unsorted_list[x + 1]
    unsorted_list[x + 1] = temp


class BubbleSort:
    @staticmethod
    def given(list):
        # print("Original list is: ", list)
        while True:
            to_last_index = len(list) - 1
            list_is_sorted = True

            for index in range(to_last_index):
                if list[index] > list[index + 1]:
                    swap(list, index)
                    list_is_sorted = False
                    # print("Sorted list is: ", list)
            # print("\n")
            if list_is_sorted:
                break
        return list

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(BubbleSort.given(INPUT_VALUE))




