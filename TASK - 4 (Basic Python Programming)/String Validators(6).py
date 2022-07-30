#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Task
# You are given a string .
# Your task is to find out if the string  contains: alphanumeric characters, alphabetical characters, digits, lowercase and uppercase characters.

# Input Format
# A single line containing a string .

# Output Format

# In the first line, print True if S has any alphanumeric characters. Otherwise, print False.
# In the second line, print True if S has any alphabetical characters. Otherwise, print False.
# In the third line, print True if S has any digits. Otherwise, print False.
# In the fourth line, print True if S has any lowercase characters. Otherwise, print False.
# In the fifth line, print True if S has any uppercase characters. Otherwise, print False.

s = input()
print(any([i.isalnum() for i in s]))
print(any([i.isalpha() for i in s]))
print(any([i.isdigit() for i in s]))
print(any([i.islower() for i in s]))
print(any([i.isupper() for i in s]))

# Note : The any() function returns True if any item in an iterable are true, otherwise it returns False. 
#        If the iterable object is empty, the any() function will return False.


# In[ ]:





# In[ ]:





# In[ ]:




