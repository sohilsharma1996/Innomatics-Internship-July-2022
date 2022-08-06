#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Transpose
# We can generate the transposition of an array using the tool numpy.transpose.
# It will not affect the original array, but it will create a new array.

# import numpy
# my_array = numpy.array([[1,2,3],
#                         [4,5,6]])
# print numpy.transpose(my_array)

# #Output
# [[1 4]
#  [2 5]
#  [3 6]]

# Flatten
# The tool flatten creates a copy of the input array flattened to one dimension.

# import numpy
# my_array = numpy.array([[1,2,3],
#                         [4,5,6]])
# print my_array.flatten()

# #Output
# [1 2 3 4 5 6]

# Task
# You are given a X integer array matrix with space separated elements ( = rows and  = columns).
# Your task is to print the transpose and flatten results.

# Input Format
# The first line contains the space separated values of  and .
# The next  lines contains the space separated elements of  columns.

# Output Format
# First, print the transpose array and then print the flatten.

# Sample Input
# 2 2
# 1 2
# 3 4

# Sample Output
# [[1 3]
#  [2 4]]
# [1 2 3 4]

import numpy

n, m = map(int, input().split())
array = numpy.array([input().strip().split() for _ in range(n)], int)
print (array.transpose())
print (array.flatten())


# In[ ]:




