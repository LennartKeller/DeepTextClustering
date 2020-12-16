import numpy as np
import pandas as pd
import requests


if __name__ == '__main__':
    r = requests.get('https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label')
    data = r.text
    texts, labels = [], []
    for line in data.split('\n'):
        tokens = line.split()
        if len(tokens) >= 2:
            l, t = tokens[0], tokens[1:]
            l = l.split(':')[0]  # we discard the fine labels
            texts.append(" ".join(t))
            labels.append(l)

    save = pd.DataFrame()
    save['texts'] = texts
    save['labels'] = labels

    save.to_csv('trec6.csv', index=False)
