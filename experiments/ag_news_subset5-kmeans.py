import os
from time import gmtime, strftime


import numpy as np
import pandas as pd
from sacred import Experiment
from sacred.observers import FileStorageObserver, MongoObserver


from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.pipeline import  Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from umap import UMAP
from transformers_clustering.helpers import purity_score, cluster_accuracy


ex = Experiment('ag_news_subset5-kmeans')
ex.observers.append(FileStorageObserver('../results/ag_news_subset5-kmeans/sacred_runs'))

mongo_enabled = os.environ.get('MONGO_SACRED_ENABLED')
mongo_user = os.environ.get('MONGO_SACRED_USER')
mongo_pass = os.environ.get('MONGO_SACRED_PASS')
mongo_host = os.environ.get('MONGO_SACRED_HOST')
mongo_port = os.environ.get('MONGO_SACRED_PORT', '27017')

if mongo_enabled == 'true':
    assert mongo_user, 'Setting $MONGO_USER is required'
    assert mongo_pass, 'Setting $MONGO_PASS is required'
    assert mongo_host, 'Setting $MONGO_HOST is required'

    mongo_url = 'mongodb://{0}:{1}@{2}:{3}/' \
                'sacred?authMechanism=SCRAM-SHA-1'.format(mongo_user, mongo_pass, mongo_host, mongo_port)

    ex.observers.append(MongoObserver(url=mongo_url, db_name='sacred'))


@ex.config
def cfg():
    n_init = 20
    max_features = 20000
    umap_n_components = 100
    dataset = "../datasets/ag_news_subset5/ag_news_subset5.csv"
    train_idx_file = "../datasets/ag_news_subset5/splits/train"
    val_idx_file = "../datasets/ag_news_subset5/splits/validation"
    result_dir = f"../results/ag_news_subset5-kmeans/{strftime('%Y-%m-%d_%H:%M:%S', gmtime())}"
    random_state = 42

@ex.automain
def run(n_init,
        max_features,
        umap_n_components,
        dataset,
        train_idx_file,
        val_idx_file,
        result_dir,
        random_state
        ):
    # Set random states
    np.random.seed(random_state)

    # load data
    df = pd.read_csv(dataset)

    with open(train_idx_file, 'r') as f:
        train_idx = np.array(list(map(int, f.readlines())))

    with open(val_idx_file, 'r') as f:
        val_idx = np.array(list(map(int, f.readlines())))

    all_idx = np.concatenate((train_idx, val_idx))

    df_train = df.iloc[all_idx].copy()

    train_texts = df_train['texts'].to_numpy()
    train_labels = df_train['labels'].to_numpy()

    df_val = df.iloc[val_idx].copy()

    val_texts = df_val['texts'].to_numpy()
    val_labels = df_val['labels'].to_numpy()

    tfidf = TfidfVectorizer(max_features=max_features, stop_words='english')
    X_train = tfidf.fit_transform(train_texts)
    X_test = tfidf.transform(val_texts)

    umap = UMAP(n_components=umap_n_components)
    X_train = umap.fit_transform(X_train.toarray())
    X_test = umap.transform(X_test.toarray())

    kmeans = KMeans(n_init=n_init, n_cluster=len(np.unique(train_labels)))
    kmeans.fit(X_train)
    predicted_labels = kmeans.predict(X_test)

    best_matching, accuracy = cluster_accuracy(val_labels, predicted_labels)
    ari = adjusted_rand_score(val_labels, predicted_labels)
    nmi = normalized_mutual_info_score(val_labels, predicted_labels)
    purity = purity_score(y_true=val_labels, y_pred=predicted_labels)

    run_results = {}
    run_results['best_matching'] = best_matching
    run_results['accuracy'] = accuracy
    run_results['ari'] = ari
    run_results['nmi'] = nmi
    run_results['purity'] = purity  # use purity to compare with microsoft paper

    os.makedirs(result_dir, exist_ok=True)
    result_df = pd.DataFrame.from_records([run_results])
    result_df.to_csv(os.path.join(result_dir, f'ag_news_subset5-kmeans.csv'), index=False)

