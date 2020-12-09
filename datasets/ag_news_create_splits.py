import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def create_split_idx(df, val_size=0.1):
    idx = np.arange(df.shape[0])
    train_idx, val_idx = train_test_split(idx, test_size=0.1, random_state=42)
    return train_idx, val_idx


if __name__ == '__main__':
    df = pd.read_csv('ag_news/ag_news.csv')
    train_idx, val_idx = create_split_idx(df)
    with open('ag_news/splits/train', 'w') as f:
        f.write("\n".join(map(str, train_idx)))
    with open('ag_news/splits/validation', 'w') as f:
        f.write("\n".join(map(str, val_idx)))

    df = pd.read_csv('ag_news_subset5/ag_news_subset5.csv')
    train_idx, val_idx = create_split_idx(df)
    with open('ag_news_subset5/splits/train', 'w') as f:
        f.write("\n".join(map(str, train_idx)))
    with open('ag_news_subset5/splits/validation', 'w') as f:
        f.write("\n".join(map(str, val_idx)))

    df = pd.read_csv('ag_news_subset10/ag_news_subset10.csv')
    train_idx, val_idx = create_split_idx(df)
    with open('ag_news_subset10/splits/train', 'w') as f:
        f.write("\n".join(map(str, train_idx)))
    with open('ag_news_subset10/splits/validation', 'w') as f:
        f.write("\n".join(map(str, val_idx)))
