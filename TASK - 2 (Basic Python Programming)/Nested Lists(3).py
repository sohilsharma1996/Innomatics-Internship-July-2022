#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Given the names and grades for each student in a class of n students, 
# store them in a nested list and print the name(s) of any student(s) having the second lowest grade.
# Note: If there are multiple students with the second lowest grade, order their names alphabetically and print each name on a new line.

n = int(input())
records = []
grade = []

for i in range(n):
    name = input()
    score = float(input())
    records.append([name,score])
    grade.append(score)                    # Calculation of the Second Lowest Score
# print("1. List of all Students with their [Names , Marks]:",records)
# print("2. List of Marks for All Students:",grade)
grade = sorted(set(grade))                 # Set will return Unique Elements. And sorted function will return Sorted List
# print("3. List of Marks for All Students(Sorted and Duplicates Removed):",grade)
m = grade[1]                               # Second lowest Grade
# print("4. Second Lowest Marks within All Students:",m)
name = []                                  # Another list to get all the names for students with Same Grades
for val in records:                        # Here , val represents all the Sub-lists inside res[]
    if m == val[1]:
        name.append(val[0])
# print("5. List of Names for All Students",name)                              # Unsorted List for Names
name.sort()
# print("6. List of Names for All Students(Sorted)",name)                      # Sorted List for Names
for nm in name:
    print(nm)                              # Print the Names in Alphabetical Order
    
#     5                      <=====  Input Values
# Harry
# 37.21
# Berry
# 37.21
# Tina
# 37.2
# Akriti
# 41
# Harsh
# 39


# In[ ]:





# In[ ]:




