import os
from time import gmtime, strftime


import numpy as np
import pandas as pd
from sacred import Experiment
from sacred.observers import FileStorageObserver, MongoObserver


from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from umap import UMAP
from transformers_clustering.helpers import purity_score, cluster_accuracy


ex = Experiment('trec6-kmeans')
ex.observers.append(FileStorageObserver('../results/sacred_runs/trec6-kmeans/'))

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
    dataset = "../datasets/trec6/trec6.csv"
    result_dir = f"../results/trec6-kmeans/{strftime('%Y-%m-%d_%H:%M:%S', gmtime())}"
    random_state = 42
    do_umap = True

@ex.automain
def run(n_init,
        max_features,
        umap_n_components,
        dataset,
        result_dir,
        random_state,
        do_umap
        ):
    # Set random states
    np.random.seed(random_state)

    # load data
    train_df = pd.read_csv(dataset)

    texts = train_df['texts'].to_numpy()
    labels = train_df['labels'].to_numpy()

    le = LabelEncoder()
    labels = le.fit_transform(labels)

    tfidf = TfidfVectorizer(max_features=max_features, stop_words='english')
    X_train = tfidf.fit_transform(texts)

    if do_umap:
        umap = UMAP(n_components=umap_n_components)
        X_train = umap.fit_transform(X_train.toarray())

    kmeans = KMeans(n_init=n_init, n_clusters=len(np.unique(labels)))
    predicted_labels = kmeans.fit_predict(X_train)

    best_matching, accuracy = cluster_accuracy(labels, predicted_labels)
    ari = adjusted_rand_score(labels, predicted_labels)
    nmi = normalized_mutual_info_score(labels, predicted_labels)
    purity = purity_score(y_true=labels, y_pred=predicted_labels)

    run_results = {}
    run_results['best_matching'] = best_matching
    run_results['accuracy'] = accuracy
    run_results['ari'] = ari
    run_results['nmi'] = nmi
    run_results['purity'] = purity  # use purity to compare with microsoft paper

    os.makedirs(result_dir, exist_ok=True)
    result_df = pd.DataFrame.from_records([run_results])
    result_df.to_csv(os.path.join(result_dir, f'trec6-kmeans.csv'), index=False)

