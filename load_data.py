import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

def get_data():
    train_df = pd.read_csv("data/public_train_trx.csv")
    test_df = pd.read_csv("data/public_test_trx.csv")
    alldf = pd.concat([train_df,test_df])
    alldf = alldf.sort_values(['session_id','duration_of_session','click_num'])
    alldf = alldf.reset_index(drop=True)
    target = 'TARGET_successful_purchase'
    
    alldf = alldf.fillna(-1)
    aggregalando_valtozok = list(alldf.columns)
    aggregalando_valtozok.remove('test_or_train_flag')
    aggregalando_valtozok.remove(target)
    aggregalando_valtozok.remove('start_date_of_session')
    aggregalando_valtozok.remove('start_time_of_session')
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