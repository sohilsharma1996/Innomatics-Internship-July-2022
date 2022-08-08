#!/usr/bin/env python
# coding: utf-8
Task
Andrea has a simple equation:

    Y  = a + b1.f1 + b1.f2 + ... + bm.fm
    
for (m+1) real constants (a,f1,f2,...,fm). We can say that the value of Y depends on m features. 
Andrea studies this equation for n different feature sets (f1,f2,...,fm) and records each respective value of Y. 
If she has q new feature sets, can you help Andrea find the value of Y for each of the sets?
Note: You are not expected to account for bias and variance trade-offs.

Input Format

The first line contains 2 space-separated integers, m (the number of observed features) and n (the number of feature sets Andrea studied), respectively.
Each of the n subsequent lines contain (m+1) space-separated decimals; the first m elements are features (f1,f2,...,fm), and the last element is the value of Y for the line's feature set.
The next line contains a single integer, q , denoting the number of feature sets Andrea wants to query for.
Each of the q subsequent lines contains m space-separated decimals describing the feature sets.

Scoring
For each feature set in one test case, we will compute the following:
    
    1) di' = (| Computed Value of Y - Expected Value of Y |) / Expected Value of Y
    2) di = max(di' - 0.1 , 0). We will permit up to a  margin of error.
    3) si = max(1.0 - di , 0)
    
The normalized score for each test case will be: S = sum(si) / q (sum from i = 1 to i = q).
If the challenge is worth C points, then your score will be S * C.

Output Format
For each of the q feature sets, print the value of Y on a new line (i.e., you must print a total of q lines).

Sample Input
2 7
0.18 0.89 109.85
1.0 0.26 155.72
0.92 0.11 137.66
0.07 0.37 76.17
0.85 0.16 139.75
0.99 0.41 162.6
0.87 0.47 151.77
4
0.49 0.18
0.57 0.83
0.56 0.64
0.76 0.18

Sample Output
105.22
142.68
132.94
129.71

Explanation
We're given m = 2, so Y = a + b1.f1 + b2.f2. We're also given n = 7.
We use the information above to find the values of a, b1, and b2. Then, we find the value of Y for each of the q feature sets.
# In[ ]:


from sklearn import linear_model

m,n = [int(x) for x in input().strip().split(' ')]
dat = [[float(x) for x in input().strip().split(' ')] for i in range(n)]

q = int(input().strip())
inp = [[float(x) for x in input().strip().split(' ')] for i in range(q)]

x = [[dat[j][i] for i in range(m)] for j in range(n)]
y = [ dat[j][m] for j in range(n)]

lm = linear_model.LinearRegression()
lm.fit(x,y)

a = lm.intercept_
b = lm.coef_

res = [a + sum([b[i] * inp[j][i] for i in range(m)]) for j in range(q)]
for y in res:
    print(y)

