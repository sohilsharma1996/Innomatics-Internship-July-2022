#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# You are given a string and your task is to swap cases. 
# In other words, convert all lowercase letters to uppercase letters and vice versa.

# Function Description

# Complete the swap_case function in the editor below.
# swap_case has the following parameters:
# string s: the string to modify

# Returns

# string: the modified string

# Input Format

# A single line containing a string s.

def swap_case(s):
    x = ""
    for c in s:
        if c.islower():
            c = c.upper()
        else:
            c = c.lower()
        x += "".join(c) 
    return x

s = input()
result = swap_case(s)
print(result)

# Another Method is to use swapcase() function as follows:
# def swap_case(s):
#    return s.swapcase()

