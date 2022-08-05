#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# group()
# A group() expression returns one or more subgroups of the match.
# Code

# >>> import re
# >>> m = re.match(r'(\w+)@(\w+)\.(\w+)','username@hackerrank.com')
# >>> m.group(0)       # The entire match 
# 'username@hackerrank.com'
# >>> m.group(1)       # The first parenthesized subgroup.
# 'username'
# >>> m.group(2)       # The second parenthesized subgroup.
# 'hackerrank'
# >>> m.group(3)       # The third parenthesized subgroup.
# 'com'
# >>> m.group(1,2,3)   # Multiple arguments give us a tuple.
# ('username', 'hackerrank', 'com')

# groups()
# A groups() expression returns a tuple containing all the subgroups of the match.
# Code

# >>> import re
# >>> m = re.match(r'(\w+)@(\w+)\.(\w+)','username@hackerrank.com')
# >>> m.groups()
# ('username', 'hackerrank', 'com')

# groupdict()
# A groupdict() expression returns a dictionary containing all the named subgroups of the match, keyed by the subgroup name.
# Code

# >>> m = re.match(r'(?P<user>\w+)@(?P<website>\w+)\.(?P<extension>\w+)','myname@hackerrank.com')
# >>> m.groupdict()
# {'website': 'hackerrank', 'user': 'myname', 'extension': 'com'}

# Task
# You are given a string S.
# Your task is to find the first occurrence of an alphanumeric character in S (read from left to right) that has consecutive repetitions.

# Input Format
# A single line of input containing the string S.

# Output Format
# Print the first occurrence of the repeating character. If there are no repeating characters, print -1.

import re
s = input()
pattern = r'([a-z A-Z 0-9])\1'
m = re.search(pattern,s)

if m:
    print(m.groups()[0])
else:
    print(-1)
    
# Note : Convention for any Pattern : r'([])\1'

