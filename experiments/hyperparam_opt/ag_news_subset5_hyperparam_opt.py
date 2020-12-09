import os
import pickle
from time import gmtime, strftime

import numpy as np
import pandas as pd
import torch
from sacred import Experiment
from sacred.observers import FileStorageObserver, MongoObserver
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import get_linear_schedule_with_warmup

from transformers_clustering.helpers import TextDataset
from transformers_clustering.model import init_model, train, concat_cls_n_hidden_states

ex = Experiment('ag_news-distilbert')
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
    n_epochs = 20
    lr = 2e-5
    batch_size = 16
    val_batch_size = 32
    base_model = "distilbert-base-uncased"
    clustering_loss_weight = 1.0
    embedding_extractor = concat_cls_n_hidden_states
    annealing_alphas = np.arange(1, n_epochs + 1)
    dataset = "../datasets/ag_news.csv"
    result_dir = f"../results/ag_news-distilbert/{strftime('%Y-%m-%d_%H:%M:%S', gmtime())}"
    train_idx_file = "../datasets/ag_news_subset5/splits/train"
    val_idx_file = "../datasets/ag_news_subset5/splits/val"
    early_stopping = True
    early_stopping_tol = 0.01
    device = "cuda:0"
    random_state = 42


@ex.automain
def run(n_epochs,
        lr,
        batch_size,
        val_batch_size,
        base_model,
        clustering_loss_weight,
        embedding_extractor,
        annealing_alphas,
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

    all_idx = np.appen(train_idx, val_idx)

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

    # init lm model & tokenizer
    lm_model = AutoModelForMaskedLM.from_pretrained(base_model, return_dict=True, output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model, return_dict=True, output_hidden_states=True)

    lm_model.to(device)

    # init clustering model
    model, initial_centroids, initial_embeddings = init_model(
        lm_model=lm_model,
        tokenizer=tokenizer,
        data_loader=train_data_loader,
        embedding_extractor=embedding_extractor,
        n_clusters=np.unique(train_labels).shape[0],
        device=device
    )

    # init optimizer & scheduler
    opt = torch.optim.RMSprop(
        params=model.parameters(),
        lr=lr,  # 2e-5, 5e-7,
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
        annealing_alphas=annealing_alphas,
        train_data_loader=train_data_loader,
        eval_data_loader=val_data_loader,
        clustering_loss_weight=clustering_loss_weight,
        early_stopping=early_stopping,
        early_stopping_tol=early_stopping_tol,
        verbose=True
    )

    # save results & model
    os.makedirs(result_dir)
    with open(os.path.join(result_dir, 'train_hist.h'), 'wb') as f:
        pickle.dump(hist, file=f)

    torch.save(model, os.path.join(result_dir, 'model.bin'))
