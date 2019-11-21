#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score

import pandas as pd
from sklearn.linear_model import LinearRegression


# In[2]:


df = pd.read_csv('../data/DataSet_Hitelbiralat_preprocessed.csv')


# In[3]:


def rounding_score_decorator(score):
    return lambda y_true, y_pred: score(y_true, y_pred > 0.5)

def text2score(optimalization):
    if optimalization == 'AUC':
        score = roc_auc_score
    elif optimalization == 'Precision':
        score = rounding_score_decorator(precision_score)
    elif optimalization == 'Recall':
        score = rounding_score_decorator(recall_score)
    elif optimalization == 'Accuracy':
        score = rounding_score_decorator(accuracy_score)
    return score

def modell_evaluator(data, input_attributes, target_attribute, model, optimalization):
    score = text2score(optimalization)
    
    split_idx = len(df) // 2
    data_train = data[:split_idx]
    data_test = data[split_idx:]
    def test_attributes(fix_input, possible_inputs):
        best_score = -1
        best_input = None
        for possible_input in possible_inputs:
            model.fit(data_train[fix_input + [possible_input]], data_train[target_attribute])
            predicted = model.predict(data_test[fix_input + [possible_input]])
            s = score(data_test[target_attribute], predicted)
            if s > best_score:
                best_score = s
                best_input = possible_input
        return best_input, best_score
    good_inputs = []
    in_race_inputs = input_attributes
    best_s = -1
    while len(in_race_inputs):
        i_to_accept, s = test_attributes([], input_attributes)
        if s < best_s:
            return best_s, good_inputs
        
        best_s = s
        good_inputs.append(i_to_accept)
        in_race_inputs.remove(i_to_accept)
    return best_s, good_inputs


# In[4]:


i = df.columns.to_list()
i.remove('TARGET_LABEL_BAD')
modell_evaluator(df, 
                 i, #['Sex', 'Age', 'MONTHS_IN_THE_JOB', 'PERSONAL_NET_INCOME', 'PAYMENT_DAY'], 
                 'TARGET_LABEL_BAD', 
                 LinearRegression(), 
                 'AUC')

