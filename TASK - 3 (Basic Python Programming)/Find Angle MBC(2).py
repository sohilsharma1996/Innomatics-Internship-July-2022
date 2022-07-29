#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ABC is a right triangle, 90 degrees at B.
# Therefore,angle ABC = 90 degrees.
# Point M is the midpoint of hypotenuse AC.
# You are given the lengths AB and BC.
# Your task is to find MBC(angle THETA) in degrees.

# Input Format

# The first line contains the length of side AB.
# The second line contains the length of side BC.

# Output Format

# Output THETA in degrees.
# Note: Round the angle to the nearest integer.

import math

AB = int(input())
BC = int(input())

print(round(math.degrees(math.atan(AB/BC))),chr(176),sep='')

# 1. chr(176) - Used to get the Degrees Sign(°).
# 2. As per calculations , AC = 2MB, and AC = 2MC. Thus , MB = MC. So Angle MBC = Angle MCB(Angle ACB) [THETA].
#    Thus , tan(THETA) = AB/BC.
# 3. .atan() = Used to get tan-1 for any Value.


# In[ ]:





# In[ ]:




