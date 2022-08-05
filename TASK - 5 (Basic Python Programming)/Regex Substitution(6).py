#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# The re.sub() tool (sub stands for substitution) evaluates a pattern and, for each valid match, it calls a method (or lambda).
# The method is called for all matches and can be used to modify strings in different ways.
# The re.sub() method returns the modified string as an output.
# Learn more about re.sub().

# Transformation of Strings
# Code

# import re

# #Squaring numbers
# def square(match):
#     number = int(match.group(0))
#     return str(number**2)

# print re.sub(r"\d+", square, "1 2 3 4 5 6 7 8 9")

# Output
# 1 4 9 16 25 36 49 64 81

# Replacements in Strings
# Code

# import re

# html = """
# <head>
# <title>HTML</title>
# </head>
# <object type="application/x-flash" 
#   data="your-file.swf" 
#   width="0" height="0">
#   <!-- <param name="movie"  value="your-file.swf" /> -->
#   <param name="quality" value="high"/>
# </object>
# """

# print re.sub("(<!--.*?-->)", "", html) #remove comment

# Output
# <head>
# <title>HTML</title>
# </head>
# <object type="application/x-flash" 
#   data="your-file.swf" 
#   width="0" height="0">

#   <param name="quality" value="high"/>
# </object>

# Task
# You are given a text of N lines. The text contains && and || symbols.
# Your task is to modify those symbols to the following:

# && → and
# || → or
# Both && and || should have a space " " on both sides.

# Input Format
# The first line contains the integer, N.
# The next N lines each contain a line of the text.

# Note : Neither && nor || occur in the start or end of each line.

# Output Format
# Output the modified text.

import re

pattern = r"(?<= )(&&|\|\|)(?= )"
replacement = lambda x: "and" if x.group() == "&&" else "or"

for i in range(int(input())):    # int(input()) denotes the no of times we want to run the entire loop
    s = input()
    print(re.sub(pattern, replacement, s))


# In[ ]:





# In[ ]:




