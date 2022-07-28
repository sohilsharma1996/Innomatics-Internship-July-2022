#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Ms. Gabriel Williams is a botany professor at District College. 
# One day, she asked her student Mickey to compute the average of all the plants with distinct heights in her greenhouse.

# Complete the average function in the editor below.
# Average has the following parameters:
# int arr: an array of integers
# Returns
# float: the resulting float value rounded to 3 places after the decimal

def average(array): 
    array = set(array)
    return sum(array)/len(array) 

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    result = average(arr)
    print(result)


# In[ ]:




