#!/usr/bin/env python
# coding: utf-8
Task
In a certain plant, the time taken to assemble a car is a random variable, X, having a normal distribution with a mean of 20 hours and a standard deviation of 2 hours. 
What is the probability that a car can be assembled at this plant in:

a) Less than 19.5 hours?
b) Between 20 and 22 hours?

Input Format
There are 3 lines of input (shown below):

20 2
19.5
20 22

The first line contains 2 space-separated values denoting the respective mean and standard deviation for X. 
The second line contains the number associated with question 1. 
The third line contains 2 space-separated values describing the respective lower and upper range boundaries for question 2.

If you do not wish to read this information from stdin, you can hard-code it into your program.

Output Format
There are two lines of output. Your answers must be rounded to a scale of 3 decimal places (i.e., 1.234 format):

On the first line, print the answer to question 1 (i.e., the probability that a car can be assembled in less than 19.5 hours).
On the second line, print the answer to question 2 (i.e., the probability that a car can be assembled in between 20 to 22 hours).
# In[ ]:


from math import sqrt,erf

def normcom(x,m,s):
    z = (x-m)/(s*sqrt(2))
    return (1 + erf(z))/2

m,s = list(map(int,input().split(" ")))
x = float(input())
x1,x2 = list(map(int,input().split(" ")))

print(round(normcom(x,m,s),3))
print(round(normcom(x2,m,s) - normcom(x1,m,s),3))


# In[ ]:




