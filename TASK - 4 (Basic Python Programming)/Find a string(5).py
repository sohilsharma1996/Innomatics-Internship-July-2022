#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# In this challenge, the user enters a string and a substring.
# You have to print the number of times that the substring occurs in the given string. 
# String traversal will take place from left to right, not from right to left.
# NOTE: String letters are case-sensitive.

# Input Format

# The first line of input contains the original string. The next line contains the substring.
# Each character in the string is an ascii character.

# Output Format
# Output the integer number indicating the total number of occurrences of the substring in the original string.

string = input()
sub_string = input()

def count_substring(string, sub_string):
    count = 0
    ml = len(string)
    sl = len(sub_string)
    for i in range(ml-sl+1):
        if (string[i:(i+sl)] == sub_string):
            count +=1
    return count

count = count_substring(string, sub_string)
print(count)


# In[ ]:





# In[ ]:





# In[ ]:




