#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Consider a list (list = []). You can perform the following commands:

# insert i e: Insert integer i at position e.
# print: Print the list.
# remove e: Delete the first occurrence of integer e.
# append e: Insert integer e at the end of the list.
# sort: Sort the list.
# pop: Pop the last element from the list.
# reverse: Reverse the list.

# Initialize your list and read in the value of n followed by n lines of commands where each command will be of the 7 types listed above. 
# Iterate through each command in order and perform the corresponding operation on your list.

list = []
n = int(input())

for i in range(n):
    command = input().split()      #converting string into list
    if command[0] == 'insert':
        list.insert(int(command[1]),int(command[2]))
    elif command[0] == 'print':
        print(list)
    elif command[0] == 'remove':
        list.remove(int(command[1]))
    elif command[0] == 'append':
        list.append(int(command[1]))
    elif command[0] == 'sort':
        list.sort()
    elif command[0] == 'pop':
        list.pop()
    else:
        list.reverse()
    

