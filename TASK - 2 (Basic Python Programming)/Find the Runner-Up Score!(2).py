#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Given the participants' score sheet for your University Sports Day, you are required to find the runner-up score. 
# You are given n scores. Store them in a list and find the score of the runner-up.
# Input Format - The first line contains n. The second line contains an array A[] of n integers each separated by a space.
# Output Format - Print the runner-up score.

n = int(input())
arr = list(map(int, input().split()))

arr.sort()
c = max(arr)
d = arr.index(c) - 1
print(arr[d])


# In[ ]:




