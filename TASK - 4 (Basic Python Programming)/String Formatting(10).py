#!/usr/bin/env python
# coding: utf-8

# In[11]:


# Given an integer, n , print the following values for each integer i from 1 to n:

# Decimal
# Octal
# Hexadecimal (capitalized)
# Binary

# Function Description

# Complete the print_formatted function in the editor below.
# print_formatted has the following parameters:
# int number: the maximum value to print

# Prints

# The four values must be printed on a single line in the order specified above for each i from 1 to number. 
# Each value should be space-padded to match the width of the binary value of number and the values should be separated by a single space.

# Input Format

# A single integer denoting n.

# Sample Input
# 17

# Sample Output

#     1     1     1     1
#     2     2     2    10
#     3     3     3    11
#     4     4     4   100
#     5     5     5   101
#     6     6     6   110
#     7     7     7   111
#     8    10     8  1000
#     9    11     9  1001
#    10    12     A  1010
#    11    13     B  1011
#    12    14     C  1100
#    13    15     D  1101
#    14    16     E  1110
#    15    17     F  1111
#    16    20    10 10000
#    17    21    11 10001

number = int(input())
l = 0

def print_formatted(number):
    l = len("{0:b}".format(number))
    
for i in range(1,number+1):
    print("{0:{w}d}   {0:{w}o}   {0:{w}X}   {0:{w}b}".format(i,w=l))

print_formatted(number)

# Note:
# 1) d - decimal values , o - octal values , X - hexadecimal values , b - binary values.
# 2) {w} - Provides the amount of space to be present between ssubsequent Rows.


# In[ ]:




