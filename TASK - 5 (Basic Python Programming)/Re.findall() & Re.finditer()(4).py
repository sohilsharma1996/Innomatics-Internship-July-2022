#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# re.findall()
# The expression re.findall() returns all the non-overlapping matches of patterns in a string as a list of strings.
# Code

# >>> import re
# >>> re.findall(r'\w','http://www.hackerrank.com/')
# ['h', 't', 't', 'p', 'w', 'w', 'w', 'h', 'a', 'c', 'k', 'e', 'r', 'r', 'a', 'n', 'k', 'c', 'o', 'm']

# re.finditer()
# The expression re.finditer() returns an iterator yielding MatchObject instances over all non-overlapping matches for the re pattern in the string.
# Code

# >>> import re
# >>> re.finditer(r'\w','http://www.hackerrank.com/')
# <callable-iterator object at 0x0266C790>
# >>> map(lambda x: x.group(),re.finditer(r'\w','http://www.hackerrank.com/'))
# ['h', 't', 't', 'p', 'w', 'w', 'w', 'h', 'a', 'c', 'k', 'e', 'r', 'r', 'a', 'n', 'k', 'c', 'o', 'm']

# Task
# You are given a string S. It consists of alphanumeric characters, spaces and symbols(+,-).
# Your task is to find all the substrings of S that contains 2 or more vowels.
# Also, these substrings must lie in between 2 consonants and should contain vowels only.

# Note :
# Vowels are defined as: AEIOU and aeiou.
# Consonants are defined as: QWRTYPSDFGHJKLZXCVBNM and qwrtypsdfghjklzxcvbnm.

# Input Format
# A single line of input containing string S.

# Output Format
# Print the matched substrings in their order of occurrence on separate lines.
# If no match is found, print -1.

import re
s = input()
v = "aeiou"
c = "bcdfghjklmnpqrstvwxyz"
l = re.findall(r"(?<=[%s])([%s]{2,})[%s]" % (c , v , c) , s , re.IGNORECASE)

if not l:
    print(-1)
else:
    for i in l:
        print(i)

# Note : ?<= matches if the current position in the string is preceded by a match
# for ... that ends at the current position
# This is called a Positive Lookbehind Assertion.
# (?<=abc)def will find a match in abcdef,
# since the lookbehind will back up 3 characters and check
# if the contained pattern matches.


# In[ ]:




