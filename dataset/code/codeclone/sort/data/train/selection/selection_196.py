#! /usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from tkinter import *


def select(vector):
	for i in range(0,len(vector)-1):
		aux = i
		for j in range(i+1,len(vector)):
			if vector[j] < vector[aux]:
				aux = j

		aux2 = vector[aux]
		vector[aux] = vector[i]
		vector[i] = aux2

	return vector
INPUT_VALUE = [5, 1, 3, 2, 1, 5, 7, 8, 10, 2]
print(select(INPUT_VALUE))