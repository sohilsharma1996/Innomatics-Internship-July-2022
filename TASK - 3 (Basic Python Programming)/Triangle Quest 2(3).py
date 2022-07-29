#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# You are given a positive integer N.
# Your task is to print a palindromic triangle of size N.
# For example, a palindromic triangle of size 5 is:

# 1
# 121
# 12321
# 1234321
# 123454321

# You can't take more than two lines. The first line (a for-statement) is already written for you.
# You have to complete the code using exactly one print statement.

# Note:
# Using anything related to strings will give a score of 0.
# Using more than one for-statement will give a score of 0.

# Input Format

# A single line of input containing the integer N.

# Output Format

# Print the palindromic triangle of size N as explained above.

for i in range(1,int(input())+1):
    print(((10 ** i) // 9) ** 2)
    
#     Note : 
#     1) 1 = 1*2 ; 121 = 11**2 ; 12321 = 111**2 and so on.
#     2) 10//9 = 1 ; 100//9 = 11 ; 1000//9 = 111, and so on.


# In[ ]:





# In[ ]:




