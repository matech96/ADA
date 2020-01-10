import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

def get_data(imputer=None, imp_columns=None):
    alldf = get_df()
    if (imputer is None) or (imp_columns is None):
        alldf = alldf.fillna(0)
    else:
#         ['basket_element_number', 'click_num',
#                        'customer_age', 'customer_value', 'duration_of_session',
#                        'last_order_of_customer', 'level_of_purchasing_process',
#                        'lifetime_customer_account', 'max_val',
#                        'maximum_price_of_visited_products',
#                        'minimum_price_of_visited_products', 'num_of_previous_payments',
#                        'price_of_cheapest_product_in_basket',
#                        'price_of_more_expensive_product_in_basket', 'regio_of_customer',
#                        'start_date_of_session', 'start_time_of_session',
#                        'sum_price_of_products_in_basket', 'sum_price_of_visited_products',
#                        'test_or_train_flag']
        alldf[imp_columns] = imputer.fit_transform(alldf[imp_columns])
        alldf = alldf.fillna(0)
    return dg2data(alldf)

def get_df():
    train_df = pd.read_csv("data/public_train_trx.csv")
    test_df = pd.read_csv("data/public_test_trx.csv")
    alldf = pd.concat([train_df,test_df])
    alldf = alldf.sort_values(['session_id','duration_of_session','click_num'])
    alldf['start_date_of_session'] = pd.to_datetime(alldf['start_date_of_session']).astype(np.int64)
    alldf['start_time_of_session'] = pd.to_datetime(alldf['start_time_of_session']).astype(np.int64)
    return alldf.reset_index(drop=True)

def dg2data(alldf):
    target = 'TARGET_successful_purchase'
    aggregalando_valtozok = list(alldf.columns)
    aggregalando_valtozok.remove('test_or_train_flag')
    aggregalando_valtozok.remove(target)
    aggregalando_valtozok.remove('session_id')
    
    cust_df = alldf.groupby('session_id',as_index=False).agg({target:'min',
                                            'test_or_train_flag':'min'})
    for aggregalos_modszer in ['min','max','mean']:
        task={}
        ujoszlonevek=[]
        for v in aggregalando_valtozok:
            task[v]=aggregalos_modszer
            ujoszlonevek.append(aggregalos_modszer+"_"+v)
        stat = alldf.groupby(['session_id'],as_index=False).agg(task)
        stat.columns=['session_id']+ujoszlonevek
        cust_df = cust_df.merge(stat,on='session_id',how='left')
        
    bemeno_valtozok = list(cust_df.columns)[3:]
    ismert_df = cust_df[ cust_df['test_or_train_flag']==0].copy()
    x_train = ismert_df[bemeno_valtozok]
    y_train = ismert_df[target]
    x_pred = cust_df[cust_df['test_or_train_flag'] == 1][bemeno_valtozok]
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_pred = scaler.transform(x_pred)
    print(f"Training set size: {x_train.shape}")
    
    return x_train, y_train, x_pred