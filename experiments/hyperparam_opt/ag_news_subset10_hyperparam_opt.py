import os
import pickle
from functools import partial
from pprint import pprint
from time import gmtime, strftime

import numpy as np
import pandas as pd
import torch
from sacred import Experiment
from sacred.observers import FileStorageObserver, MongoObserver
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import ParameterGrid
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import get_linear_schedule_with_warmup

from transformers_clustering.helpers import TextDataset, orig_annealing_alphas
from transformers_clustering.helpers import cluster_accuracy, purity_score
from transformers_clustering.model import (
    init_model,
    train,
    evaluate,
    concat_cls_n_hidden_states,
    concat_mean_n_hidden_states
)

ex = Experiment('ag_news_subset10_opt-distilbert')
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
    n_epochs = 5  # we increase the number of epoch
    # we use the best params from subset5 run and only tune the lr
    hyperparam_grid = {
        'clustering_loss_weight': [1.0],
        'annealing_alphas': [np.ones(n_epochs) * 1000.0],
        'embedding_extractor':
            [partial(concat_cls_n_hidden_states, n=2)],
        'lr': sorted([2e-4, 1e-4, 3e-5, 5e-5, 2e-5, 1e-5, 2e-6, 5e-6, 7e-5, 9e-5, 9e-6, 9e-5])
    }
    batch_size = 16
    val_batch_size = 32
    base_model = "distilbert-base-uncased"
    embedding_extractor = concat_cls_n_hidden_states
    annealing_alphas = np.arange(1, n_epochs + 1)
    dataset = "../datasets/ag_news_subset10/ag_news_subset10.csv"
    result_dir = f"../results/ag_news_subset10-distilbert/opt/{strftime('%Y-%m-%d_%H:%M:%S', gmtime())}"
    train_idx_file = "../datasets/ag_news_subset10/splits/train"
    val_idx_file = "../datasets/ag_news_subset10/splits/validation"
    early_stopping = True
    early_stopping_tol = 0.01
    device = "cuda:0"
    random_state = 42


@ex.automain
def run(n_epochs,
        hyperparam_grid,
        batch_size,
        val_batch_size,
        base_model,
        dataset,
        train_idx_file,
        val_idx_file,
        result_dir,
        early_stopping,
        early_stopping_tol,
        device,
        random_state
        ):
    # Set random states
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)

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

    train_data = TextDataset(train_texts, train_labels)
    train_data_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)

    df_val = df.iloc[val_idx]
    val_texts = df_val['texts'].to_numpy()
    val_labels = df_val['labels'].to_numpy()

    val_data = TextDataset(val_texts, val_labels)
    val_data_loader = DataLoader(dataset=val_data, batch_size=val_batch_size, shuffle=False)

    # insert code here!
    results = []
    param_grid = ParameterGrid(hyperparam_grid)
    for run_idx, params in enumerate(param_grid):
        print(f'Run: {run_idx+1}/{len(list(param_grid))}')
        print("Running with params:")
        pprint(params)

        # init lm model & tokenizer
        lm_model = AutoModelForMaskedLM.from_pretrained(base_model, return_dict=True, output_hidden_states=True)
        tokenizer = AutoTokenizer.from_pretrained(base_model, return_dict=True, output_hidden_states=True)

        lm_model.to(device)

        # init clustering model
        model, initial_centroids, initial_embeddings = init_model(
            lm_model=lm_model,
            tokenizer=tokenizer,
            data_loader=train_data_loader,
            embedding_extractor=params['embedding_extractor'],
            n_clusters=np.unique(train_labels).shape[0],
            device=device
        )

        # init optimizer & scheduler
        opt = torch.optim.RMSprop(
            params=model.parameters(),
            lr=params['lr'],  # hier weitermachen
            eps=1e-8
        )

        total_steps = len(train_data_loader) * n_epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer=opt,
            num_warmup_steps=int(len(train_data_loader) * 0.5),
            num_training_steps=total_steps
        )

        # train the model
        hist = train(
            n_epochs=n_epochs,
            model=model,
            optimizer=opt,
            scheduler=scheduler,
            annealing_alphas=params['annealing_alphas'],
            train_data_loader=train_data_loader,
            eval_data_loader=val_data_loader,
            clustering_loss_weight=params['clustering_loss_weight'],
            early_stopping=early_stopping,
            early_stopping_tol=early_stopping_tol,
            verbose=True
        )

        # do eval
        run_results = {**{f'param_{key}': value for key, value in params.items()}}

        predicted_labels, true_labels = evaluate(
            model=model,
            eval_data_loader=val_data_loader,
            verbose=True
        )

        best_matching, accuracy = cluster_accuracy(true_labels, predicted_labels)
        ari = adjusted_rand_score(true_labels, predicted_labels)
        nmi = normalized_mutual_info_score(true_labels, predicted_labels)
        purity = purity_score(y_true=true_labels, y_pred=predicted_labels)

        run_results['best_matching'] = best_matching
        run_results['accuracy'] = accuracy
        run_results['ari'] = ari
        run_results['nmi'] = nmi


        # save train hist
        os.makedirs(result_dir, exist_ok=True)

        results.append(run_results)
        result_df = pd.DataFrame.from_records(results)
        result_df.to_csv(os.path.join(result_dir, 'opt_results_ag_news_subset10.csv'), index=False)

        with open(os.path.join(result_dir, f'train_hist_run{run_idx}.h'), 'wb') as f:
            pickle.dump(hist, file=f)
