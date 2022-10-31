

def shell_sort(list):
    length = len(list)
    # 希尔排序length=len(seq)
    num = 0
    while num <= length / 3:
        num = num * 3 + 1
    while num >= 1:
        for i in range(num,length):
            tmp = list[i]
            for j in range(i,0,- num):
                if tmp < list[j - num]:
                    list[j] = list[j - num]
                else:
                    j += num
                    break
            list[j - num] = tmp
        num //= 3
    print(list)


#时间复杂度：O(n^1.3)
#空间复杂度：0(1)
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(shell_sort(INPUT_VALUE))