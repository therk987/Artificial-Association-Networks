#---------------------------------------------------------------------------------------------------------------
# Switch to insertion sort if the length of the array is SMALLNESS_THRESHOLD or less.

from array import *

def median3(a, b, c):
  # Find the median of a, b, c.
  if (a < b):
    if (b < c):
      m = b
    elif (a < c):
      m = c
    else:
      m = a
  else:
    if (b > c):
      m = b
    elif (a > c):
      m = c
    else:
      m = a

  return m
#---------------------------------------------------------------------------------------------------------------
def quicksort (x):
    # Sort a numeric Python array.  This is what the user calls.
    n = len(x)
    return qsort(x, 0, n-1)
#---------------------------------------------------------------------------------------------------------------
def qsort(x, left, right):
    QUICKSORT_SMALLNESS_THRESHOLD = 7
    if (right <= left):
        return

    if (left + QUICKSORT_SMALLNESS_THRESHOLD <= right):
        # Compute the pivot using Tukey's median-of-median-of-three pseudomedian idea.
        stride = (right - left)//8
        med1  = median3(x[left],          x[left+stride],   x[left+2*stride])
        med2  = median3(x[left+3*stride], x[left+4*stride], x[left+5*stride])
        med3  = median3(x[left+6*stride], x[left+7*stride], x[right])
        pivot = median3(med1, med2, med3)

        #print ("pivot: ",pivot, type(pivot))
        # Use a three-way partition that moves values equal to the pivot to the left and right ends
        # of the block being partitioned.
        a = b = left
        c = d = right
        while (True):
            while (b <= c) and (x[b] <= pivot):
                if (x[b] == pivot):
                    x[a], x[b], = x[b], x[a]
                    a += 1
                b += 1

            while (c >= b) and (x[c] >= pivot):
                if (x[c] == pivot):
                    x[c], x[d] = x[d], x[c]
                    d -= 1
                c -= 1

            if (b > c):
                break
            else:
                x[b], x[c] = x[c], x[b]
                b += 1
                c -= 1

        # Now move values equal to the pivot to the middle of the block.
        s = min(a-left, b-a)
        l = left
        h = b-s
        while (s > 0):
            x[l], x[h] = x[h], x[l]
            s -= 1
            l += 1
            h += 1

        s = min(d-c, right-d)
        l = b
        h = right+1-s
        while (s > 0):
            x[l], x[h] = x[h], x[l]
            s -= 1
            l += 1
            h += 1

        # Apply quicksort recursively to the left and right ends of the block.
        qsort(x, left, left+(b-a))
        qsort(x, (right+1)-(d-c), right)
    else: # Switch to insertion sort for small sorts.
        for i in range(left+1, right+1):
            for j in range(i, left, -1):
                if (x[j] < x[j-1]):
                    x[j], x[j-1] = x[j-1], x[j]  # Swap out-of-order terms.
    return x



INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(quicksort(INPUT_VALUE))