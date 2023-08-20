import pandas as pd
# create a function to take a categorical column and return the one hot encoded version of it
# input: dataframe, column name
# output: dataframe with one hot encoded column

def one_hot_encode(df, column_name):
    one_hot_encoded = pd.get_dummies(df[column_name], prefix=column_name)
    df = df.drop(column_name, axis=1)
    df = df.join(one_hot_encoded)
    return df
