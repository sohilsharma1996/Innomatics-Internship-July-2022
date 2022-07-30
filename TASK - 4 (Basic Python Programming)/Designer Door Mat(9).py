#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Mr. Vincent works in a door mat manufacturing company. 
# One day, he designed a new door mat with the following specifications:

# Mat size must be N x M. (N is an odd natural number, and M is 3 times N.)
# The design should have 'WELCOME' written in the center.
# The design pattern should only use | . and - characters.
# Sample Designs

#     Size: 7 x 21 
#     ---------.|.---------
#     ------.|..|..|.------
#     ---.|..|..|..|..|.---
#     -------WELCOME-------
#     ---.|..|..|..|..|.---
#     ------.|..|..|.------
#     ---------.|.---------
    
#     Size: 11 x 33
#     ---------------.|.---------------
#     ------------.|..|..|.------------
#     ---------.|..|..|..|..|.---------
#     ------.|..|..|..|..|..|..|.------
#     ---.|..|..|..|..|..|..|..|..|.---
#     -------------WELCOME-------------
#     ---.|..|..|..|..|..|..|..|..|.---
#     ------.|..|..|..|..|..|..|.------
#     ---------.|..|..|..|..|.---------
#     ------------.|..|..|.------------
#     ---------------.|.---------------
    
# Input Format
# A single line containing the space separated values of N and M.

# Output Format
# Output the design pattern.

R,C = map(int,input().split())

for i in range(1,R,2):
    print((".|."*i).center(C,'-'))
    
print("WELCOME".center(C,'-'))

for i in range(R-2,-1,-2):
    print((".|."*i).center(C,'-'))


# In[ ]:





# In[ ]:




