import os
import pickle
from time import gmtime, strftime

import numpy as np
import pandas as pd
import torch
from sacred import Experiment
from sacred.observers import FileStorageObserver
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import get_linear_schedule_with_warmup

from transformers_clustering.helpers import TextDataset
from transformers_clustering.model import init_model, train, concat_cls_n_hidden_states

ex = Experiment('ag_news_subset5-distilbert')
ex.observers.append(FileStorageObserver('../results/sacred_runs'))


@ex.config
def cfg():
    n_epochs = 5
    lr = 2e-5
    batch_size = 16
    base_model = "distilbert-base-uncased"
    clustering_loss_weight = 0.1
    embedding_extractor = concat_cls_n_hidden_states
    annealing_alphas = np.arange(1, n_epochs + 1)
    dataset = "../datasets/ag_news_subset5.csv"
    result_dir = f"../results/ag_news_subset5-distilbert/{strftime('%Y-%m-%d_%H:%M:%S', gmtime())}"
    val_size = 0.1  # not used
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
        dataset,
        result_dir,
        val_size,
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
        verbose=True
    )

    # save results
    os.mkdir(result_dir)
    with open(os.path.join(result_dir, 'train_hist.h'), 'wb') as f:
        pickle.dump(hist, file=f)
