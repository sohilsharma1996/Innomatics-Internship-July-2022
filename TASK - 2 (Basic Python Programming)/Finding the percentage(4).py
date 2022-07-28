#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# The provided code stub will read in a dictionary containing key/value pairs of name:[marks] for a list of students. 
# Print the average of the marks array for the student name provided, showing 2 places after the decimal.

n = int(input())
student_marks = {}     # Dictionary

for i in range(n):
    name, *line = input().split()    # *line converts the Inputs provided for line variable into a List of String Numbers
    scores = list(map(float, line))  # Converting values in line into float datatype
    student_marks[name] = scores     # passing name as key and scores as value
query_name = input()
marks = student_marks[query_name]    # Storing marks for any Query Name(getting values for corresponding key in any Dictionary)

print(format(sum(marks)/3,'.2f'))
# Format Function Syntax = format(value,'.nf')

# 3                           <<============= Input Values
# Krishna 67 68 69
# Arjun 70 98 63
# Malika 52 56 60
# Malika


# In[ ]:





# In[ ]:




