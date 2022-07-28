#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Given 2 sets of integers, M and N, print their symmetric difference in ascending order. 
# The term symmetric difference indicates those values that exist in either M or N but do not exist in both.

# #          Input Format
# The first line of input contains an integer, .
# The second line contains  space-separated integers.
# The third line contains an integer, .
# The fourth line contains  space-separated integers.
# #          Output Format
# Output the symmetric difference integers in ascending order, one per line.

n = int(input())
s1 = set(map(int,input().split()))
m = int(input())
s2 = set(map(int,input().split()))

s3 = s1.union(s2)

l = []
for i in s3:
    if i in s1 and i in s2:
        continue
    else:
        l.append(i)

l.sort()

for i in l:
    print(i)


# In[ ]:




