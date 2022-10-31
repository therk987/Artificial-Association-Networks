#Author: Denis Karanja, P15/55431/2012
#Institution: The University of Nairobi, Kenya,
#Department: School of Computing and Informatics, Chiromo campus
#Email: dee.caranja@gmail.com,
#Task: Merge sort- divides a list into 3 parts in Python
#License type: MIT :)


def divideList(fullList):
    length = len(fullList)
    if length <= 1:
      return fullList

    else:
        #divide elemnts
        midOne = length // 3
        midTwo = ((length - midOne) * -1) // 2

        if midOne is not length:	
            #divide list into three parts
            listOne = fullList[0:midOne]
            listTwo = fullList[midOne:midTwo]
            listThree = fullList[midTwo:]

            #conquer lists
            sortedListOne = divideList(listOne)
            sortedListTwo = divideList(listTwo)
            sortedListThree = divideList(listThree)

            return mergeLists(sortedListOne, sortedListTwo, sortedListThree)

        else:
            return "\nThe list has to be divided :)"
def mergeTwoLists(listOne, listTwo):
    i = j = 0
    output = []
    lengthOne = len(listOne)
    lengthTwo = len(listTwo)

    #deal with two lists first
    while i < lengthOne or j < lengthTwo:
        #case 1 (both lists have data)
        if i < lengthOne and j < lengthTwo:
            if listOne[i] <= listTwo[j]:
                output += [listOne[i]]
                i += 1
            else:
                output += [listTwo[j]]
                j += 1

        #case 2 (only listOne have data)
        elif i < lengthOne:
            output += [listOne[i]]
            i += 1

            #case 3 (only list two has data)
        else:
            output += [listTwo[j]]
            j += 1

    return output

def mergeLists(listOne, listTwo, listThree):
    output = outputFinal = []

    #merge listOne and listTwo
    output = mergeTwoLists(listOne, listTwo)

    #Merge listThree and (output = listOne+listTwo)
    outputFinal = mergeTwoLists(output, listThree)


    return outputFinal

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(divideList(INPUT_VALUE))