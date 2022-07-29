#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# You are given a complex z. Your task is to convert it to polar coordinates.

# Input Format

# A single line containing the complex number z. 
# Note: complex() function can be used in python to convert the input as a complex number.

# Output Format

# Output two lines:
# The first line should contain the value of r.
# The second line should contain the value of phi.

import cmath

# .strip() returns a copy of the string in which all chars have been stripped from the beginning and the end of the string.

c = complex(input().strip())  # Input a Complex Number
res = cmath.polar(c)          # .polar() converts Complex no from Rectangular Coordinates to Polar Coordinates
print(res[0])
print(res[1])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




