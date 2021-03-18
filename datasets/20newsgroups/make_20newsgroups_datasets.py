import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

if __name__ == '__main__':
    # Fetch the dataset
    data = fetch_20newsgroups(subset="all")
    texts = np.array(data.data)
    labels = np.array(data.target)
    df = pd.DataFrame(data={'texts': texts, 'labels': labels})
    df.to_csv('20newsgroups.csv', index=False)
    # Read split indices
    with open('splits/test', 'r') as f:
        test_idx = np.array(list(map(int, f.read().splitlines())))
    with open('splits/validation', 'r') as f:
        validation_idx = np.array(list(map(int, f.read().splitlines())))

    assert not set(test_idx).intersection(set(validation_idx))

    test_texts, test_labels = texts[test_idx], labels[test_idx]
    val_texts, val_labels = texts[validation_idx], labels[validation_idx]

    concat_idx = np.append(test_idx, validation_idx)
    texts, labels = np.delete(texts, concat_idx), np.delete(labels, concat_idx)

    df_test = pd.DataFrame(data={'texts': test_texts, 'labels': test_labels})
    df_val = pd.DataFrame(data={'texts': val_texts, 'labels': val_labels})

    # the test set contains is also used for training because it's unsupervised learning
    df_test.to_csv('20newsgroups_train.csv', index=False)
    df_val.to_csv('20newsgroups_val.csv', index=False)
