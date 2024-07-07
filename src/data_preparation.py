import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_prepare_data(url):
    df = pd.read_csv(url)
    df = df[['comment', 'stars']]
    df['comment'] = df['comment'].astype(
        str)  # Ensure all comments are strings
    return df


def split_data(df, test_size=0.2, val_size=0.125):
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=1)
    train_df, val_df = train_test_split(
        train_df, test_size=val_size, random_state=1)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
