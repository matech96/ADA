import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import linear_model

def modell_evaluator(data, input_attributes, target_attribute, model):
    data_cloumns = input_attributes + [target_attribute]
    data = data[data_cloumns]
    data = data.dropna()
    columnes_to_binarry, columnes_to_onehot, columnes_to_remove = sort_columns(data, target_attribute)

    data = data.drop(columnes_to_remove, axis=1)

    data = columns_to_binarry(columnes_to_binarry, data)

    data = columns_to_onehot(columnes_to_onehot, data)

    x_columns = data.columns.tolist()
    x_columns.remove(target_attribute)
    x = data[x_columns]
    y = data[target_attribute]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    model.fit(x_train, y_train)

    mape = mean_absolute_percentage_error(y_test, model.predict(x_test))

    eq_str = get_eq_str(data, model)

    return mape, eq_str


def sort_columns(data, target_attribute):
    columnes_to_binarry = []
    columnes_to_onehot = []
    columnes_to_remove = []
    for index, value in data.dtypes.items():
        if value == object:
            n_values = len(data[index].unique())
            if n_values == 1:
                columnes_to_remove.append(index)
            if n_values == 2:
                columnes_to_binarry.append(index)
            elif n_values > 2:
                columnes_to_onehot.append(index)
    assert target_attribute not in columnes_to_onehot, 'The target variable can not be multicategorical'
    return columnes_to_binarry, columnes_to_onehot, columnes_to_remove


def columns_to_binarry(columnes_to_binarry, data):
    for column in columnes_to_binarry:
        v = data[column].unique()[0]
        data[column] = data[column].apply(lambda x: 1 if x == v else 0)
    return data


def columns_to_onehot(columnes_to_onehot, data):
    for column in columnes_to_onehot:
        one_hot = pd.get_dummies(data[column])
        one_hot.columns = one_hot.columns + '_' + column
        data = data.drop(column, axis=1)
        data = data.join(one_hot)
    return data


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_idx = y_true != 0
    return np.mean(np.abs((y_true - y_pred)[non_zero_idx] / y_true[non_zero_idx])) * 100


def get_eq_str(data, model):
    eq = []
    for w, x in zip(model.coef_, data.columns):
        eq.append(f'({w})*({x})')
    eq_str = '+'.join(eq + [f'({model.intercept_})'])
    return eq_str