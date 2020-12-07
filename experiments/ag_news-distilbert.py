import os
import pickle
from time import gmtime, strftime

import numpy as np
import tensorflow_datasets as tfds
import torch
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.utils import shuffle
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import get_linear_schedule_with_warmup

from transformers_clustering.helpers import TextDataset
from transformers_clustering.model import init_model, train, concat_cls_n_hidden_states

ex = Experiment('ag_news-distilbert')
ex.observers.append(FileStorageObserver('../results/sacred_runs'))


@ex.config
def cfg():
    n_epochs = 20
    lr = 2e-5
    batch_size = 8
    base_model = "distilbert-base-uncased"
    clustering_loss_weight = 1.0
    embedding_extractor = concat_cls_n_hidden_states
    annealing_alphas = np.arange(1, n_epochs + 1)
    result_dir = f"../results/ag_news-distilbert/{strftime('%Y-%m-%d_%H:%M:%S', gmtime())}"
    val_size = 0.1  # not used
    early_stopping = True
    early_stopping_tol = 0.01
    device = "cuda:0"
    random_state = 42


@ex.automain
def run(n_epochs,
        lr,
        batch_size,
        base_model,
        clustering_loss_weight,
        embedding_extractor,
        annealing_alphas,
        result_dir,
        val_size,
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
    train_ds = tfds.load('ag_news_subset', split='train', shuffle_files=True)
    test_ds = tfds.load('ag_news_subset', split='test', shuffle_files=True)
    texts, labels = [], []
    for ds in (train_ds, test_ds):
        for example in tfds.as_numpy(ds):
            text, label = example['description'], example['label']
            texts.append(text.decode("utf-8"))
            labels.append(label)
    labels = np.array(labels)
    del train_ds
    del test_ds

    text, labels = shuffle(texts, labels, random_state=random_state)

    data = TextDataset(texts, labels)
    data_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=False)

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
    opt = torch.optim.RMSprop(
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
