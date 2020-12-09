import numpy as np
import pandas as pd
import tensorflow_datasets as tfds

if __name__ == '__main__':
    train_ds = tfds.load('ag_news_subset', split='train', shuffle_files=True)
    test_ds = tfds.load('ag_news_subset', split='test', shuffle_files=True)
    texts, labels = [], []
    for ds in (train_ds, test_ds):
        for example in tfds.as_numpy(ds):
            text, label = example['description'], example['label']
            texts.append(text.decode("utf-8"))
            labels.append(label)
    labels = np.array(labels)

    save = pd.DataFrame()
    save['texts'] = texts
    save['labels'] = labels

    save.to_csv('ag_news.csv', index=False)
