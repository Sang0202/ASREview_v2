# create a function to take a dataframe and return the sample of it with a number of instances to label in the next step
# input: dataframe
# output: dataframe with a number of instances to label
def sample_to_label(df, sample_size):
    df = df.sample(n=sample_size, random_state=1)
    return df

