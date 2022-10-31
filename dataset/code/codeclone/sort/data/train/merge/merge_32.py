# ?????????? ????????
# O(N * logN) ?????? ? ?????????
# ????? ??????????, ?? in-State(????? ????????????? N ??????)
# ?? ???? ????????? ??????? ??????, ???? ?? ?? ????? ????????? ?
# ????? ???????? ?????????? ??????? ??????????, ????????? ??
# ?????? ??? ?????? ???? ? ?????? ?????????? ? ???????? MIN ?? ???

def mergeSort(array):
  if len(array) > 1:
    print('len = %s ' % len(array))

    middle = len(array) // 2
    leftArray = mergeSort(array[0: middle])
    rightArray = mergeSort(array[middle: len(array)])

    print('left array', leftArray)
    print('right array', rightArray)

    # ????????? array, ???????? leftArray ? rightArray
    mergeArray(array, leftArray, rightArray)

  return array

# ?????????? ??? ??????? ? ???? ? ??????? ????????? ?????????????
def mergeArray(array, leftArray, rightArray):
  leftIndex = 0
  rightIndex = 0
  arrayIndex = 0

  while leftIndex < len(leftArray) and rightIndex < len(rightArray):
    if leftArray[leftIndex] <= rightArray[rightIndex]:
      print('l %s ' % leftArray[leftIndex])
      array[arrayIndex] = leftArray[leftIndex]
      leftIndex += 1

    else:
      print('l %s ' % rightArray[rightIndex])
      array[arrayIndex] = rightArray[rightIndex]
      rightIndex += 1

    arrayIndex += 1

  # ???? ???????? ?? ??????????? ???????? ? ????? ???????
  if leftIndex < len(leftArray):
    # add left Items
    for i in range(leftIndex, len(leftArray)):
      array[arrayIndex] = leftArray[i]
      arrayIndex += 1
  else:
    # add right Items
    for i in range(rightIndex, len(rightArray)):
      array[arrayIndex] = rightArray[i]
      arrayIndex += 1




INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(mergeSort(INPUT_VALUE))