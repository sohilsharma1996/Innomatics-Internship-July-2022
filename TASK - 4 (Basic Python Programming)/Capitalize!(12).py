#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# You are asked to ensure that the first and last names of people begin with a capital letter in their passports. 
# For example, alison heck should be capitalised correctly as Alison Heck.
# alison heck -----> Alison Heck
# Given a full name, your task is to capitalize the name appropriately.

# Input Format
# A single line of input containing the full name, S .

# Output Format
# Print the capitalized string, S .

s = input()

def solve(s):
    l = s.split(" ")
    s = ' '
    for i in l:
        s+=i.capitalize()+ ' '      # Capitalize() returns the Capital 1st Letter for any String present.
    return s

solve(s)

