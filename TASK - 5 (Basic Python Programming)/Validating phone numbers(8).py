#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Let's dive into the interesting topic of regular expressions! 
# You are given some input, and you are required to check whether they are valid mobile numbers.

# A valid mobile number is a ten digit number starting with a 7,8 or 9.

# Concept
# A valid mobile number is a ten digit number starting with a 7.8 or 9.
# Regular expressions are a key concept in any programming language.

# Input Format
# The first line contains an integer N, the number of inputs.
# N lines follow, each containing some string.
    
# Output Format
# For every string listed, print "YES" if it is a valid mobile number and "NO" if it is not on separate lines. 
# Do not print the quotes.

# Sample Input

# 2
# 9587456281
# 1252478965

# Sample Output

# YES
# NO

import re
def checker(contact):
    pattern = r"[789]\d{9}$"
    if re.match(pattern, contact):
        return "YES"
    else:
        return "NO"

n = int(input())
for i in range(n):
    print(checker(input()))

