#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Students of District College have subscriptions to English and French newspapers. 
# Some students have subscribed to English only, some have subscribed to French only, and some have subscribed to both newspapers.
# You are given two sets of student roll numbers. 
# One set has subscribed to the English newspaper, and one set has subscribed to the French newspaper. 
# Your task is to find the total number of students who have subscribed to either the English or the French newspaper but not both.

# Input Format

# The first line contains the number of students who have subscribed to the English newspaper.
# The second line contains the space separated list of student roll numbers who have subscribed to the English newspaper.
# The third line contains the number of students who have subscribed to the French newspaper.
# The fourth line contains the space separated list of student roll numbers who have subscribed to the French newspaper.

# Output Format

# Output total number of students who have subscriptions to the English or the French newspaper but not both.

e = int(input())
E = set(map(int,input().split()))
f = int(input())
F = set(map(int,input().split()))

ans = E.symmetric_difference(F)

count = 0

for i in ans:
    count+=1
print(count)


# In[ ]:




