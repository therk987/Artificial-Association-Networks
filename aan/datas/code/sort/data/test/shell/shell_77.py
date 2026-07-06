
def shell_sort(u_list):
    s_list = u_list
    l = int(len(s_list)/2)
    while l > 0:
        for i in range(l, int(len(s_list)),+1):
            temp = s_list[i]
            j = i
            while j >= l and s_list[j-l] > temp:
               s_list[j] = s_list[j-l]
               j = j-l
            s_list[j] = temp
        l = l//2
    return s_list

INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(shell_sort(INPUT_VALUE))