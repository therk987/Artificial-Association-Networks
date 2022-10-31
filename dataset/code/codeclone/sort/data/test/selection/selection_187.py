def sort(numbers): 

	for i in range(len(numbers)-1):
		mainposition = i
		for j in range(i,len(numbers)):
			if numbers[j] < numbers[mainposition]:
				mainposition = j

		temp = numbers[i]
		numbers[i] = numbers[mainposition]
		numbers[mainposition] = temp
	return numbers


INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(sort(INPUT_VALUE))