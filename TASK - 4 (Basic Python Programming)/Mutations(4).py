#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Task
# Read a given string, change the character at a given index and then print the modified string.

# Function Description
# Complete the mutate_string function in the editor below.

# mutate_string has the following parameters:

# string string: the string to change
# int position: the index to insert the character at
# string character: the character to insert

# Returns
# string: the altered string

# Input Format
# The first line contains a string , string .
# The next line contains an integer , position, the index location and a string ,character, separated by a space.

s = input()
i,c = input().split()          # Taking 2 Input Parameters Position and Characters(both of different Data TIpes)

def mutate_string(s, i, c):
    return s[:i] + c + s[i+1:]

print(mutate_string(s,int(i),c))


# In[ ]:





# In[ ]:




