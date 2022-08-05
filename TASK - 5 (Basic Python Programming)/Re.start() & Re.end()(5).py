#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# start() & end()
# These expressions return the indices of the start and end of the substring matched by the group.

# Code

# >>> import re
# >>> m = re.search(r'\d+','1234')
# >>> m.end()
# 4
# >>> m.start()
# 0

# Task
# You are given a string S.
# Your task is to find the indices of the start and end of string k in S.

# Input Format
# The first line contains the string S.
# The second line contains the string k.

# Output Format
# Print the tuple in this format: (start _index, end _index).
# If no match is found, print (-1, -1).

import re
s = input()
k = input()
pattern = re.compile(k)
m = pattern.search(s)      # Searching for pattern in string input s
if not m:
    print("(-1, -1)")
else:
    while m:
        print("({0}, {1})".format(m.start(), m.end() - 1))
        m = pattern.search(s, m.start()+ 1) 


# In[ ]:




