#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Given an integer,n, and n space-separated integers as input, create a tuple,t, of those n integers.
# Then compute and print the result of hash(t).

n = int(input())
integer_list = map(int, input().split())
print(hash(tuple(integer_list)))

