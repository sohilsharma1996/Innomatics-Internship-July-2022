#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# re
# A regular expression (or RegEx) specifies a set of strings that matches it.
# A regex is a sequence of characters that defines a search pattern, mainly for the use of string pattern matching.

# The re.search() expression scans through a string looking for the first location where the regex pattern produces a match.
# It either returns a MatchObject instance or returns None if no position in the string matches the pattern.

# Code

# >>> import re
# >>> print bool(re.search(r"ly","similarly"))
# True

# The re.match() expression only matches at the beginning of the string.
# It either returns a MatchObject instance or returns None if the string does not match the pattern.

# Code

# >>> import re
# >>> print bool(re.match(r"ly","similarly"))
# False
# >>> print bool(re.match(r"ly","ly should be in the beginning"))
# True


# TASK

# You are given a string N.
# Your task is to verify that N is a floating point number.

# In this task, a valid float number must satisfy all of the following requirements:

#  Number can start with +, - or . symbol.
# For example:
# ✔
# +4.50
# ✔
# -1.0
# ✔
# .5
# ✔
# -.7
# ✔
# +.4
# ✖
#  -+4.5

#  Number must contain at least 1 decimal value.
# For example:
# ✖
#  12.
# ✔
# 12.0  

#  Number must have exactly one . symbol.
#  Number must not give any exceptions when converted using float(N).

# Input Format

# The first line contains an integer T, the number of test cases.
# The next T line(s) contains a string .

# Output Format

# Output True or False for each test case.

import re
pattern = r'^[+-]?[0-9]*\.[0-9]+$'
n = int(input())
for i in range(n):
    s = input()
    print(bool(re.match(pattern,s)))
    
# r'^[+-]? says that Input starts with + or -.
# [0-9]* says that the next char will be 0,1,2,3,4,5,6,7,8, or 9 (* says that digit may also appear once/twice/more times)
# \ is a skip char that is used for "." , without using . , since compiler can't understand "." char
# [0-9]+$' says that Last Digit will be atleast one digit from 0,1,2,3,4,5,6,7,8 or 9 decimal digit. Dolar Sign is ending symbol. 
# The Python module re provides full support for Perl-like regular expressions in Python. 
# The re module raises the exception re. error if an error occurs while compiling or using a regular expression. 
# We would cover two important functions, which would be used to handle regular expressions.

# MATCH Function - This function attempts to match RE pattern to string with optional flags.

# Here is the syntax for this function −

# re.match(pattern, string, flags=0)
# Here is the description of the parameters −

# Parameter & Description
# 1	- Pattern
# This is the regular expression to be matched.

# 2 - String
# This is the string, which would be searched to match the pattern at the beginning of string.

# 3 - Flags
# You can specify different flags using bitwise OR (|). These are modifiers, which are listed in the table below.

