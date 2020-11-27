import pandas as pd
import numpy as np
import tensorflow_datasets as tfds
from itertools import chain
from sklearn.model_selection import train_test_split


def make_ag_news_subset(out_path: str, subset_size=0.1, random_state=42):

    """
    Create a subset of the ag_news dataset as found in tf-datasets.
    :param out_path:
    :param subset_size:
    :param random_state:
    :return:
    """
    train_ds = tfds.load('ag_news_subset', split='train', shuffle_files=True)
    test_ds = tfds.load('ag_news_subset', split='test', shuffle_files=True)

    texts, labels = [], []

    for ds in (train_ds, test_ds):
       for example in tfds.as_numpy(ds):
           text, label = example['description'], example['label']
           texts.append(text.decode("utf-8"))
           labels.append(label)

    labels = np.array(labels)

    test_subset, _, labels_subset, _ = train_test_split(texts, labels,
                                                       test_size=1-subset_size,
                                                       random_state=random_state)

    df = pd.DataFrame()
    df['texts'] = test_subset
    df['labels'] = labels_subset

    if out_path:
        df.to_csv(out_path)

    return df


if __name__ == '__main__':
    make_ag_news_subset('../../../datasets/ag_news_subset10.csv', subset_size=0.1)
    make_ag_news_subset('../../../datasets/ag_news_subset5.csv', subset_size=0.05)
