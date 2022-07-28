#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# You are given a set A and n other sets.
# Your job is to find whether set A is a strict superset of each of the N sets.
# Print True, if  is a strict superset of each of the  sets. Otherwise, print False.
# A strict superset has at least one element that does not exist in its subset.

# Input Format

# The first line contains the space separated elements of set .
# The second line contains integer , the number of other sets.
# The next  lines contains the space separated elements of the other sets.

# Output Format

# Print True if set  is a strict superset of all other  sets. Otherwise, print False.

def checker():
    sub = set(map(int,input().split()))
    if sub.issubset(a):
        if len(a) == len(sub):
            l.append(0)
        else:
            l.append(1)
    else:
        l.append(0)

a = set(map(int,input().split()))
n = int(input())
l = []

for i in range(n):
    checker()

if all(l) == 1:
    print("True")
else:
    print("False")


# In[ ]:




