#!/usr/bin/env python
# coding: utf-8

# In[11]:


def kishazifeladat1_EMP2B5(l):
    if len(l) == 0:
        return 0
    
    return sum(l[::2]) - sum(l[1::2])

