#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


def get_typespeiffic_price(df, property_subtype):
    df_panel = df[df['property_subtype'] == property_subtype]
    return df_panel[['city', 'unit_property_price']].groupby(by=['city'])['unit_property_price'].mean()

def get_statistics(filename = '../data/DataSet_LakasP.csv'):
    df = pd.read_csv(filename)
    df['unit_property_price'] = df.price_created_at / df.property_area
    df_city_gr = df[['city', 'price_created_at', 'unit_property_price']].groupby(by='city')
    mean_unit_property_price = df_city_gr['unit_property_price'].mean().values
    min_price_created_at = df_city_gr['price_created_at'].min()
    max_price_created_at = df_city_gr['price_created_at'].max()
    df_panel_price = get_typespeiffic_price(df, 'prefabricated panel flat (for sale)')
    df_brick_price = get_typespeiffic_price(df, 'brick flat (for sale)')
    df_price_diff = (df_panel_price - df_brick_price).abs()
    return pd.DataFrame({'City': df_price_diff.index,
                         'Avg. m2 price': mean_unit_property_price, 
                         'Min': min_price_created_at,
                         'Max': max_price_created_at,
                         'df_panel_price': df_panel_price,
                         'df_brick_price': df_brick_price,
                         'Panel – brick diff': df_price_diff})


# In[5]:


get_statistics()


# In[12]:


def get_data_cleaning_helper(filename = '../data/DataSet_LakasP.csv'):
    df = pd.read_csv(filename)
    assert not df.price_created_at.isnull().any()
    df['unit_property_price'] = df.price_created_at / df.property_area
    df_panel_price = get_typespeiffic_price(df, 'prefabricated panel flat (for sale)')
    df_brick_price = get_typespeiffic_price(df, 'brick flat (for sale)')
    df_price = pd.DataFrame({'unit_property_price_panel': df_panel_price, 'unit_property_price_brick': df_brick_price})

    df_null = df[df.property_subtype.isnull()]
    df_join_pr = df_null.join(df_price, on='city')
    return pd.DataFrame({'City': df_join_pr.city,
                         'Property_subtype': df_join_pr.property_subtype,
                         'Property_area': df_join_pr.property_area,
                         'Price_created_at': df_join_pr.price_created_at,
                         'Avg. m2 price': df_join_pr.unit_property_price,
                         'Avg. m2 price panel': df_join_pr.unit_property_price_panel,
                         'Avg. m2 price brick': df_join_pr.unit_property_price_brick})


# In[14]:


df = get_data_cleaning_helper()
get_diff = lambda x: (df['Avg. m2 price'] - df[x]).abs()
df['is_panel'] = get_diff('Avg. m2 price panel') < get_diff('Avg. m2 price brick')
df


# In[ ]:




