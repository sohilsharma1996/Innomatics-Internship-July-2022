#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# The textwrap module provides two convenient functions: wrap() and fill().

# a) textwrap.wrap()
# The wrap() function wraps a single paragraph in text (a string) so that every line is width characters long at most.
# It returns a list of output lines.

# >>> import textwrap
# >>> string = "This is a very very very very very long string."
# >>> print textwrap.wrap(string,8)
# ['This is', 'a very', 'very', 'very', 'very', 'very', 'long', 'string.'] 

# b) textwrap.fill()
# The fill() function wraps a single paragraph in text and returns a single string containing the wrapped paragraph.

# >>> import textwrap
# >>> string = "This is a very very very very very long string."
# >>> print textwrap.fill(string,8)
# This is
# a very
# very
# very
# very
# very
# long
# string.

# Task
# You are given a string S and width w.
# Your task is to wrap the string into a paragraph of width w.

# Function Description
# Complete the wrap function in the editor below.
# wrap has the following parameters:
# string string: a long string
# int max_width: the width to wrap to

# Returns
# string: a single string with newline characters ('\n') where the breaks should be.

# Input Format

# The first line contains a string , string.
# The second line contains the width , max_width .

# Sample Input

# ABCDEFGHIJKLIMNOQRSTUVWXYZ
# 4

# Sample Output 

# ABCD
# EFGH
# IJKL
# IMNO
# QRST
# UVWX
# YZ

import textwrap

def wrap(string, max_width):
    string, max_width = input(), int(input())
    return textwrap.fill(string,max_width)

result = wrap(string, max_width)
print(result)

