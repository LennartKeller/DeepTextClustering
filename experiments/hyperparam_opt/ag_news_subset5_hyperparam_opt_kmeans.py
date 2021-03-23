import os
from pprint import pprint
from time import gmtime, strftime

import numpy as np
import pandas as pd
from sacred import Experiment
from sacred.observers import FileStorageObserver, MongoObserver
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import ParameterGrid

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

from umap import UMAP

from transformers_clustering.helpers import cluster_accuracy


ex = Experiment('ag_news_subset5_opt-kmeans')
ex.observers.append(FileStorageObserver('../results/sacred_runs'))

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
    hyperparam_grid = {'umap__n_components': [10, 50, 100, 200, 300, 500, 1000, 2000, 3000]}
    dataset = "../datasets/ag_news_subset5/ag_news_subset5.csv"
    result_dir = f"../results/ag_news_subset5-kmeans/opt/{strftime('%Y-%m-%d_%H:%M:%S', gmtime())}"
    train_idx_file = "../datasets/ag_news_subset5/splits/train"
    val_idx_file = "../datasets/ag_news_subset5/splits/validation"
    random_state = 42


@ex.automain
def run(hyperparam_grid,
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

    all_idx = np.append(train_idx, val_idx)

    df_train = df.iloc[all_idx]
    train_texts = df_train['texts'].to_numpy()
    train_labels = df_train['labels'].to_numpy()

    df_val = df.iloc[val_idx]
    val_texts = df_val['texts'].to_numpy()
    val_labels = df_val['labels'].to_numpy()

    tfidf = TfidfVectorizer(max_features=20000, stop_words='english')
    X_train = tfidf.fit_transform(train_texts).toarray()
    X_val = tfidf.transform(val_texts).toarray()

    pipeline = Pipeline([
        ('umap', UMAP()),
        ('kmeans', KMeans(n_cluster=len(np.unique(train_labels)), n_jobs=6, n_init=20))
    ])

    # insert code here!
    results = []
    param_grid = ParameterGrid(hyperparam_grid)
    for run_idx, params in enumerate(param_grid):
        print(f'Run: {run_idx+1}/{len(list(param_grid))}')
        print("Running with params:")
        pprint(params)

        pipeline.set_params(**params)

        pipeline.fit_transform(X_train)
        predicted_labels = pipeline.predict(X_val)

        # do eval
        run_results = {**{f'param_{key}': value for key, value in params.items()}}

        best_matching, accuracy = cluster_accuracy(val_labels, predicted_labels)
        ari = adjusted_rand_score(val_labels, predicted_labels)
        nmi = normalized_mutual_info_score(val_labels, predicted_labels)

        run_results['best_matching'] = best_matching
        run_results['accuracy'] = accuracy
        run_results['ari'] = ari
        run_results['nmi'] = nmi

        # save train hist
        os.makedirs(result_dir, exist_ok=True)

        results.append(run_results)
        result_df = pd.DataFrame.from_records(results)
        result_df.to_csv(os.path.join(result_dir, 'opt_results_ag_news_subset5_kmeans.csv'), index=False)
