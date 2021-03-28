import os
import pickle
from functools import partial
from time import gmtime, strftime

import numpy as np
import pandas as pd
import torch
from sacred import Experiment
from sacred.observers import FileStorageObserver, MongoObserver
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from transformers_clustering.helpers import TextDataset, purity_score, cluster_accuracy
from transformers_clustering.model import concat_cls_n_hidden_states

ex = Experiment('ag_news_subset5-embeddings-kmeans')
ex.observers.append(FileStorageObserver('../results/ag_news_subset5-embeddings-kmeans/sacred_runs'))

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
    models = [
        'distilbert-base-uncased',
        'bert-base-uncased',
        'bert-large-uncased',
        'roberta-base',
        'sentence-transformers/bert-base-nli-cls-token',
        'sentence-transformers/distilbert-base-nli-cls-token'
    ]
    n_layers = 5
    embedding_extractor = partial(concat_cls_n_hidden_states, n=n_layers)
    batch_size = 16
    dataset = "../datasets/ag_news_subset5/ag_news_subset5.csv"
    train_idx_file = "../datasets/ag_news_subset5/splits/train"
    val_idx_file = "../datasets/ag_news_subset5/splits/validation"
    result_dir = f"../results/ag_news_subset5-embeddings-kmeans/{strftime('%Y-%m-%d_%H:%M:%S', gmtime())}"
    random_state = 42
    device = 'cuda:0'

@ex.automain
def run(n_init,
        models,
        embedding_extractor,
        batch_size,
        dataset,
        train_idx_file,
        val_idx_file,
        result_dir,
        random_state,
        device
        ):

    # Set random states
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)

    os.makedirs(result_dir, exist_ok=True)

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

    train_data = TextDataset(train_texts, train_labels)
    train_data_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)


    df_val = df.iloc[val_idx].copy()

    val_texts = df_val['texts'].to_numpy()
    val_labels = df_val['labels'].to_numpy()

    val_data = TextDataset(val_texts, val_labels)
    val_data_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)

    results = []
    for model in models:
        # init lm model & tokenizer
        lm_model = AutoModel.from_pretrained(model, return_dict=True, output_hidden_states=True)
        tokenizer = AutoTokenizer.from_pretrained(model, return_dict=True, output_hidden_states=True)
        lm_model.to(device)

        train_embeddings = []
        train_labels = []
        for batch_texts, batch_labels in tqdm(train_data_loader, desc="Extracting train embeddings"):
            inputs = tokenizer(list(batch_texts), return_tensors='pt', padding=True, truncation=True)
            inputs = inputs.to(device)
            with torch.no_grad():
                outputs = lm_model(**inputs)
            extracted_embeddings = embedding_extractor(outputs).cpu().detach().numpy()
            train_embeddings.append(extracted_embeddings)
            train_labels.extend(batch_labels.numpy().astype('int'))

        X_train = np.vstack(train_embeddings)
        train_labels = np.array(train_labels)

        test_embeddings = []
        val_labels = []
        for batch_texts, batch_labels in tqdm(val_data_loader, desc="Extracting val embeddings"):
            inputs = tokenizer(list(batch_texts), return_tensors='pt', padding=True, truncation=True)
            inputs = inputs.to(device)
            with torch.no_grad():
                outputs = lm_model(**inputs)
            extracted_embeddings = embedding_extractor(outputs).cpu().detach().numpy()
            test_embeddings.append(extracted_embeddings)
            val_labels.extend(batch_labels.numpy().astype('int'))

        X_test = np.vstack(test_embeddings)
        val_labels = np.array(val_labels)

        kmeans = KMeans(n_init=n_init, n_clusters=len(np.unique(train_labels)))
        kmeans.fit(X_train)
        predicted_labels = kmeans.predict(X_test)

        best_matching, accuracy = cluster_accuracy(val_labels, predicted_labels)
        ari = adjusted_rand_score(val_labels, predicted_labels)
        nmi = normalized_mutual_info_score(val_labels, predicted_labels)
        purity = purity_score(y_true=val_labels, y_pred=predicted_labels)

        run_results = {}
        run_results['model'] = model
        run_results['best_matching'] = best_matching
        run_results['accuracy'] = accuracy
        run_results['ari'] = ari
        run_results['nmi'] = nmi
        run_results['purity'] = purity  # use purity to compare with microsoft paper
        results.append(run_results)

        with open(os.path.join(result_dir, f'{model.sub("/", "_")}_embeddings.h'), 'wb') as f:
            pickle.dump([X_train, train_labels, X_test, val_labels], f)


    result_df = pd.DataFrame.from_records(results)
    result_df.to_csv(os.path.join(result_dir, f'ag_news_subset5-embeddings-kmeans.csv'), index=False)

