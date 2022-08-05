#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# re.split()
# The re.split() expression splits the string by occurrence of a pattern.

# Code
# >>> import re
# >>> re.split(r"-","+91-011-2711-1111")
# Output => ['+91', '011', '2711', '1111']

# You are given a string S consisting only of digits 0-9, commas ,, and dots .
# Your task is to complete the regex_pattern defined below, which will be used to re.split() all of the , and . symbols in S.
# It’s guaranteed that every comma and every dot in S is preceeded and followed by a digit.

# Note : In Regex Pattern = "[,.]+" , if by chance we have 2 adjacent commas in any string, then the output also 
# includes "". To removw this, + sign is used in this Statement , at the end of [,.]. Otherwise, + not needed
# Also, if there are multiple parameters to consider in re.split, then we can put all of them in [] , without using commas.

regex_pattern = r"[,.]+"

import re
print("\n".join(re.split(regex_pattern, input())))

# Note : In Regex Pattern = "[,.]+" , if by chance we have 2 adjacent commas in any string, then the output also 
# includes "". To removw this, + sign is used in this Statement , at the end of [,.]. Otherwise, + not needed
# Also, if there are multiple parameters to consider in re.split, then we can put all of them in [] , without using commas.


# In[ ]:




