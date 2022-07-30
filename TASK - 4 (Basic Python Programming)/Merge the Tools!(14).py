#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Link to the Complete Question : https://www.hackerrank.com/challenges/merge-the-tools/problem?isFullScreen=true

# Complete the merge_the_tools function in the editor below.
# merge_the_tools has the following parameters:
# string s: the string to analyze
# int k: the size of substrings to analyze

# Prints
# Print each subsequence on a new line. There will be (n/k) of them. No return value is expected.

# Input Format
# The first line contains a single string, s .
# The second line contains an integer, k , the length of each substring.

# Sample Input
# STDIN       Function
# -----       --------
# AABCAAADA   s = 'AABCAAADA'
# 3           k = 3

# Sample Output
# AB
# CA
# AD

string, k = input(), int(input())

def merge_the_tools(string, k):
    c = 0
    s = ''
    for i in string:
        if i not in s:
            s = s + i
        c += 1
        if c == k:
            print(s)
            c = 0   # Initializing the String s , thats why count back to Zero
            s = ''

merge_the_tools(string, k)


# In[ ]:




