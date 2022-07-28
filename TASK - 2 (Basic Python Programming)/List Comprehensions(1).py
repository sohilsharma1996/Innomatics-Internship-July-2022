#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# You are given three integers x,y and z representing the dimensions of a cuboid along with an integer n. 
# Print a list of all possible coordinates given by (i,j,k) on a 3D grid where the sum of i+j+k is not equal to n. 
# Here , 0<=i<=x , 0<=j<=y , 0<=k<=z . Please use list comprehensions rather than multiple loops, as a learning exercise.

x = int(input())
y = int(input())
z = int(input())
n = int(input())

list = []
XYZ = []

for i in range(x+1):
    for j in range(y+1):
        for k in range(z+1):
            if i+j+k != n:
                XYZ = [i,j,k]
                list.append(XYZ)
print(list, end="")

#Another Way to solve using List Comprehension:

# output = [[X,Y,Z] for X in range(x+1) for Y in range(y+1) for Z in range(z+1) if X+Y+Z != n]
# print (output)


# In[ ]:





# In[ ]:




