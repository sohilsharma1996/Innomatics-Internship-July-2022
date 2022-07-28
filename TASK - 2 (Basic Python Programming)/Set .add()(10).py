#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Apply your knowledge of the .add() operation to help your friend Rupal.
# Rupal has a huge collection of country stamps. She decided to count the total number of distinct country stamps in her collection. 
# She asked for your help. You pick the stamps one by one from a stack of N country stamps.
# Find the total number of distinct country stamps.

# Input Format
# The first line contains an integer N, the total number of country stamps.
# The next N lines contains the name of the country where the stamp is from.

# Output Format
# Output the total number of distinct country stamps on a single line.

no_of_stamps = int(input())
stamps = set()

for stamp in range(no_of_stamps):
    stamps.add(input())    # Input from User is to be added into the SET Stamps

print(len(stamps))


# In[ ]:





# In[ ]:




