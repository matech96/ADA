#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import Counter, defaultdict


# In[2]:


def kishazifeladat2_EMP2B5(init_list, list_to_convert):
        element_counts = Counter(init_list).most_common()
        if len(init_list) == 0:
            element2index = {}
        else:
            slice_count = element_counts[min(len(element_counts), 10) - 1][1]
            element2index = {}
            for element in init_list:
                if element not in element2index:
                    element2index[element] = len(element2index)
            for element, count in element_counts:
                if count < slice_count:
                    element2index[element] = -1
        return [element2index[element] if element in element2index else -1 for element in list_to_convert]

