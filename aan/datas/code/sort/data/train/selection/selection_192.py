
class SelectionSort(object):

    def swap(self, array, i, j):
        temp = array[i]
        array[i] = array[j]
        array[j] = temp

    def sort(self, array):
        for i in range(len(array) - 1): # -1 ?.?. ?? ????? ????????? last
            index = i

            for j in range(i + 1, len(array), 1):
                if array[index] > array[j]: # ???? min item ????? i
                    index = j

            if index != i:
                self.swap(array, index, i)

        return array


selectionSort = SelectionSort()
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]

print(selectionSort.sort(INPUT_VALUE))