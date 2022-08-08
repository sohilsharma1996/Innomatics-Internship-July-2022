#!/usr/bin/env python
# coding: utf-8
Task
The final grades for a Physics exam taken by a large group of students have a mean of 70 and a standard deviation of 10. 
If we can approximate the distribution of these grades by a normal distribution, what percentage of the students:

Scored higher than 80 (i.e., have a grade >= 80)?
Passed the test (i.e., have a grade >= 60)?
Failed the test (i.e., have a grade < 60)?

Find and print the answer to each question on a new line, rounded to a scale of 2 decimal places.

Input Format
There are 3 lines of input (shown below):

70 10
80
60

The first line contains 2 space-separated values denoting the respective mean and standard deviation for the exam. 
The second line contains the number associated with question 1. 
The third line contains the pass/fail threshold number associated with questions 2 and 3.

If you do not wish to read this information from stdin, you can hard-code it into your program.

Output Format
There are three lines of output. Your answers must be rounded to a scale of 2 decimal places (i.e., 1.23 format):

On the first line, print the answer to question 1 (i.e., the percentage of students having a grade >= 80).
On the second line, print the answer to question 2 (i.e., the percentage of students having a grade >= 60).
On the third line, print the answer to question 3 (i.e., the percentage of students having a grade < 60).
# In[ ]:


import math

mean,std = list(map(int,input().split(" ")))
x1 = float(input())
x2 = float(input())

cdf = lambda x: 0.5 * (1 + math.erf((x - mean) / (std * (2 ** 0.5))))

print(round((1-cdf(x1)) * 100 , 2))
print(round((1-cdf(x2)) * 100 , 2))
print(round((cdf(x2)) * 100 , 2))

