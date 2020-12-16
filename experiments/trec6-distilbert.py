import os
import pickle
from functools import partial
from time import gmtime, strftime

import numpy as np
import pandas as pd
import torch
from sacred import Experiment
from sacred.observers import FileStorageObserver, MongoObserver
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import get_linear_schedule_with_warmup

from transformers_clustering.helpers import TextDataset
from transformers_clustering.helpers import cluster_accuracy, purity_score
from transformers_clustering.model import init_model, train, concat_cls_n_hidden_states, evaluate

ex = Experiment('trec6-distilbert')
ex.observers.append(FileStorageObserver('../results/trec6-distilbert/sacred_runs'))

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
    n_epochs = 10
    lr = 2e-5
    train_batch_size = 8
    base_model = "distilbert-base-uncased"
    clustering_loss_weight = 1.0
    embedding_extractor = partial(concat_cls_n_hidden_states, n=2)
    annealing_alphas = np.ones(n_epochs) * 1000
    dataset = "../datasets/trec6/trec6.csv"
    result_dir = f"../results/trec6-distilbert/{strftime('%Y-%m-%d_%H:%M:%S', gmtime())}"
    early_stopping = True
    early_stopping_tol = 0.01
    device = "cuda:0"
    random_state = 42


@ex.automain
def run(n_epochs,
        lr,
        train_batch_size,
        base_model,
        clustering_loss_weight,
        embedding_extractor,
        annealing_alphas,
        dataset,
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



    texts = df['texts'].to_numpy()
    labels = df['labels'].to_numpy()

    le = LabelEncoder()
    labels_enc = le.fit_transform(labels)

    data = TextDataset(texts, labels_enc)
    data_loader = DataLoader(dataset=data, batch_size=train_batch_size, shuffle=False)


    # init lm model & tokenizer
    lm_model = AutoModelForMaskedLM.from_pretrained(base_model, return_dict=True, output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model, return_dict=True, output_hidden_states=True)

    lm_model.to(device)

    # init clustering model
    model, initial_centroids, initial_embeddings = init_model(
        lm_model=lm_model,
        tokenizer=tokenizer,
        data_loader=data_loader,
        embedding_extractor=embedding_extractor,
        n_clusters=np.unique(labels).shape[0],
        device=device
    )

    # init optimizer & scheduler
    opt = torch.optim.AdamW(
        params=model.parameters(),
        lr=lr,  # 2e-5, 5e-7,
        eps=1e-8
    )

    total_steps = len(data_loader) * n_epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer=opt,
        num_warmup_steps=int(len(data_loader) * 0.5),
        num_training_steps=total_steps
    )

    # train the model
    hist = train(
        n_epochs=n_epochs,
        model=model,
        optimizer=opt,
        scheduler=scheduler,
        annealing_alphas=annealing_alphas,
        train_data_loader=data_loader,
        eval_data_loader=data_loader,
        clustering_loss_weight=clustering_loss_weight,
        early_stopping=early_stopping,
        early_stopping_tol=early_stopping_tol,
        verbose=True
    )

    # do eval
    run_results = {}

    predicted_labels, true_labels = evaluate(
        model=model,
        eval_data_loader=data_loader,
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
    run_results['purity'] = purity  # use purity to compare with microsoft paper

    # save train hist
    os.makedirs(result_dir, exist_ok=True)

    result_df = pd.DataFrame.from_records([run_results])
    result_df.to_csv(os.path.join(result_dir, 'trec6-distilbert.csv'), index=False)

    # save results & model
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, 'train_hist.h'), 'wb') as f:
        pickle.dump(hist, file=f)

    torch.save(model, os.path.join(result_dir, 'model.bin'))
