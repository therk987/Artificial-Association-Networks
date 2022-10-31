# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 15:23:49 2017

@author: wattai
"""

# ShellSortを実行時のステップ数と実行時間の観点から評価するプログラム．
# 実行にとても時間(10分くらい)がかかります．注意．

import numpy as np
def shellsort(a):
    n = len(a)
    n_step = 0

    h = 1
    while h < n/9:
        h = 3*h + 1

    while h > 0:
        print("h: ", h)
        for i in range(h, n):
            w = a[i]
            j = i-h
            while w < a[j] and j >= 0:
                a[j+h] = a[j]
                n_step += 1  # step数のカウント
                j -= h
            a[j+h] = w
        h = np.int(np.floor(h / 3))
    return a, n_step


INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(shellsort(INPUT_VALUE))