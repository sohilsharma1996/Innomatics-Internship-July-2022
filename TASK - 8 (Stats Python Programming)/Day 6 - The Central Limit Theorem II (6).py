#!/usr/bin/env python
# coding: utf-8
Task
The number of tickets purchased by each student for the University X vs. University Y football game follows a distribution that has a mean of 2.4 and a standard deviation of 2.0.

A few hours before the game starts, 100 eager students line up to purchase last-minute tickets. 
If there are only 250 tickets left, what is the probability that all 100 students will be able to purchase tickets?

Input Format
There are 4 lines of input (shown below):

250
100
2.4
2.0

The first line contains the number of last-minute tickets available at the box office. 
The second line contains the number of students waiting to buy tickets. 
The third line contains the mean number of purchased tickets, and the fourth line contains the standard deviation.

If you do not wish to read this information from stdin, you can hard-code it into your program.

Output Format
Print the probability that 100 students can successfully purchase the remaining 250 tickets, rounded to a scale of 4 decimal places (i.e., 1.2345 format).
# In[ ]:


from math import erf,sqrt

tickets = int(input())
students = int(input())
mean = float(input())
sigma = float(input())

def cdf(x,mu,sig):
    return ( 1 + erf((x - mu)/ sqrt(2) / sig )) / 2

mu = mean * students
sig = sigma * sqrt(students)

print(round(cdf(tickets,mu,sig) , 4 ))

