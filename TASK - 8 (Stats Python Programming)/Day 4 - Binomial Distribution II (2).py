#!/usr/bin/env python
# coding: utf-8
Task
A manufacturer of metal pistons finds that, on average, 12% of the pistons they manufacture are rejected because they are incorrectly sized. 
What is the probability that a batch of 10 pistons will contain:

    a) No more than 2 rejects?
    b) At least 2 rejects?
    
Input Format
A single line containing the following values (denoting the respective percentage of defective pistons and the size of the current batch of pistons):

12 10

If you do not wish to read this information from stdin, you can hard-code it into your program.

Output Format
Print the answer to each question on its own line:

a) The first line should contain the probability that a batch of 10 pistons will contain no more than 2 rejects.
b) The second line should contain the probability that a batch of 10 pistons will contain at least 2 rejects.

Round both of your answers to a scale of 3 decimal places (i.e., 1.234 format).
# In[ ]:


def fact(n):
    return 1 if n == 0 else n*fact(n-1)

def comb(n, x):
    return fact(n) / (fact(x) * fact(n-x))

def b(x, n, p):
    return comb(n, x) * p**x * (1-p)**(n-x)

p, n = list(map(int, input().split(" ")))
print(round(sum([b(i, n, p/100) for i in range(3)]), 3))
print(round(sum([b(i, n, p/100) for i in range(2, n+1)]), 3))


# In[ ]:




