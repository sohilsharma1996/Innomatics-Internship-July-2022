#!/usr/bin/env python
# coding: utf-8
Task
A large elevator can transport a maximum of 9800 pounds. 
Suppose a load of cargo containing 49 boxes must be transported via the elevator. 
The box weight of this type of cargo follows a distribution with a mean of 205 pounds and a standard deviation of 15 pounds. 
Based on this information, what is the probability that all 49 boxes can be safely loaded into the freight elevator and transported?

Input Format
There are 4 lines of input (shown below):

9800
49
205
15

The first line contains the maximum weight the elevator can transport. 
The second line contains the number of boxes in the cargo. 
The third line contains the mean weight of a cargo box, and the fourth line contains its standard deviation.

If you do not wish to read this information from stdin, you can hard-code it into your program.

Output Format
Print the probability that the elevator can successfully transport all 49 boxes, rounded to a scale of 4 decimal places (i.e., 1.2345 format).
# In[ ]:


import math

x = int(input())
n = int(input())
mu = int(input())
sigma = int(input())

mu_sum = n * mu

sigma_sum = math.sqrt(n) * sigma

def cdf(x,mu,sigma):
    Z = (x - mu)/sigma
    return 0.5*(1 + math.erf(Z/(math.sqrt(2))))

print(round(cdf(x , mu_sum , sigma_sum), 4))

