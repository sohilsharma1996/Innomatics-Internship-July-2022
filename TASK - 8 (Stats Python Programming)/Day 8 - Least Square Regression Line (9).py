#!/usr/bin/env python
# coding: utf-8
Task
A group of five students enrolls in Statistics immediately after taking a Math aptitude test. 
Each student's Math aptitude test score, x , and Statistics course grade, y , can be expressed as the following list of (x,y) points:

(95 , 85)
(85 , 95)
(80 , 70)
(70 , 65)
(60 , 70)

If a student scored an 80 on the Math aptitude test, what grade would we expect them to achieve in Statistics? 
Determine the equation of the best-fit line using the least squares method, then compute and print the value of y when x = 80.

Input Format
There are five lines of input; each line contains two space-separated integers describing a student's respective x and y grades:

95 85
85 95
80 70
70 65
60 70

If you do not wish to read this information from stdin, you can hard-code it into your program.

Output Format
Print a single line denoting the answer, rounded to a scale of 3 decimal places (i.e., 1.234 format).
# In[ ]:


n = int(input())
x1 = int(input())

xy = [map(int,input().split()) for i in range(n)]
sx, sy, sx2, sxy = map(sum, zip(*[(x, y, x**2, x * y) for x, y in xy]))
b = (n * sxy - sx * sy) / (n * sx2 - sx**2)
a = (sy / n) - b * (sx / n)
print('{:.3f}'.format(a + b * x1))

