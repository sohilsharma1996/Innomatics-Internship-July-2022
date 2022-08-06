#!/usr/bin/env python
# coding: utf-8

# In[2]:


# zeros
# The zeros tool returns a new array with a given shape and type filled with 's.

# import numpy
# print numpy.zeros((1,2))                    #Default type is float

# Output : [[ 0.  0.]] 

# print numpy.zeros((1,2), dtype = numpy.int) #Type changes to int

# #Output : [[0 0]]

# ones
# The ones tool returns a new array with a given shape and type filled with 's.

# import numpy
# print numpy.ones((1,2))                    #Default type is float

# #Output : [[ 1.  1.]] 

# print numpy.ones((1,2), dtype = numpy.int) #Type changes to int
# #Output : [[1 1]]

# Task
# You are given the shape of the array in the form of space-separated integers, each integer representing the size of different dimensions, your task is to print an array of the given shape and integer type using the tools numpy.zeros and numpy.ones.

# Input Format
# A single line containing the space-separated integers.

# Output Format
# First, print the array using the numpy.zeros tool and then print the array with the numpy.ones tool.

# Sample Input 0
# 3 3 3

# Sample Output 0

# [[[0 0 0]
#   [0 0 0]
#   [0 0 0]]

#  [[0 0 0]
#   [0 0 0]
#   [0 0 0]]

#  [[0 0 0]
#   [0 0 0]
#   [0 0 0]]]
# [[[1 1 1]
#   [1 1 1]
#   [1 1 1]]

#  [[1 1 1]
#   [1 1 1]
#   [1 1 1]]

#  [[1 1 1]
#   [1 1 1]
#   [1 1 1]]]

# Explanation 0
# Print the array built using numpy.zeros and numpy.ones tools and you get the result as shown.

import numpy

nums = tuple(map(int, input().split()))
print (numpy.zeros(nums, dtype = numpy.int64))
print (numpy.ones(nums, dtype = numpy.int64))


# In[ ]:




