#!/usr/bin/env python
# coding: utf-8
Task
Given two n-element data sets, X and Y , calculate the value of the Pearson correlation coefficient.

Input Format
The first line contains an integer, n , denoting the size of data sets X and Y.
The second line contains n space-separated real numbers (scaled to at most one decimal place), defining data set X.
The third line contains n space-separated real numbers (scaled to at most one decimal place), defining data set Y.

Constraints
1<= xi <= 500, where xi is the ith value of data set X.
1<= yi <= 500, where yi is the ith value of data set Y.
Data set X contains unique values.
Data set Y contains unique values.

Output Format
Print the value of the Pearson correlation coefficient, rounded to a scale of 3 decimal places.

Sample Input
10
10 9.8 8 7.8 7.7 7 6 5 4 2 
200 44 32 24 22 17 15 12 8 4

Sample Output
0.612

Explanation
The mean and standard deviation of data set X are: 6.73(mu_x) , 2.39251(sigma_x)
The mean and standard deviation of data set Y are: 37.8(mu_y) , 55.1993(sigma_y)

We use the following formula to calculate the Pearson correlation coefficient:
    
    P(X,Y) = SUM [[(xi - mu_x) * (yi - mu_y)] / (n * sigma_x * sigma_y)]
# In[ ]:


N = int(input())
X = list(map(float,input().strip().split()))
Y = list(map(float,input().strip().split()))

mu_x = sum(X) / N
mu_y = sum(Y) / N

std_x = (sum([(i - mu_x)**2 for i in X]) / N)**0.5
std_y = (sum([(i - mu_y)**2 for i in Y]) / N)**0.5

covariance = sum([(X[i] - mu_x) * (Y[i] - mu_y) for i in range(N)])

correlation_coefficient = covariance / (N * std_x * std_y)

print(round(correlation_coefficient,3))

