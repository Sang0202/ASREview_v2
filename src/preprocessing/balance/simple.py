# create a function that does not resample the data
# input: dataframe
# output: dataframe
#
def simple_sample(X, y):
    return X,y

# # create a function that undersamples the data
# # input: dataframe with a label column
# # output: dataframe

# def undersample(df):
#     smallest_class_size = df['label'].value_counts().min()
#     df = df.groupby('label').apply(lambda x: x.sample(n=smallest_class_size, random_state=1))
#     return df

# # create a function that finds the smallest class size then return the dataframe with the smallest class size sample of each class
# # input: dataframe with a label column
# # output: dataframe

# def undersample_all(df):
#     smallest_class_size = df['label'].value_counts().min()
#     df = df.groupby('label').apply(lambda x: x.sample(n=smallest_class_size, random_state=1))
#     return df




# # create a function that oversamples the data
# # input: dataframe
# # output: dataframe

# def oversample(df):
#     df = df.groupby('label').apply(lambda x: x.sample(n=100, replace=True, random_state=1))
#     return df
