from copy import deepcopy
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import torch
from sklearn.cluster._kmeans import _k_init
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.utils.extmath import row_norms
from torch import nn
from tqdm import tqdm
from transformers.file_utils import ModelOutput

from .helpers import cluster_accuracy
from .helpers import lp_distance
from .helpers import mask_tokens


@dataclass
class ClusterOutput(ModelOutput):
    loss: torch.FloatTensor = None
    predicted_labels: torch.IntTensor = None
    embeddings: torch.FloatTensor = None


def cls_embedding_extractor(model_output: ModelOutput):
    return model_output.last_hidden_state[:, 0, :].float()


def meanpooler_embedding_extractor(model_ouput: ModelOutput):
    return model_ouput.last_hidden_state.mean(dim=1).float()


def concat_cls_n_hidden_states(model_output: ModelOutput, n=3):
    n_hidden_states = model_output.hidden_states[-n:]
    return torch.cat([t[:, 0, :] for t in n_hidden_states], 1).float()


def concat_mean_n_hidden_states(model_output: ModelOutput, n=3):
    n_hidden_states = model_output.hidden_states[-n:]
    return torch.cat([t.mean(dim=1) for t in n_hidden_states], 1).float()


class LearnableWeightedAverage(nn.Module):

    def __init__(self, n=3, device='cuda:0'):
        super(LearnableWeightedAverage, self).__init__()
        self.n = n
        self.device = device
        self.register_parameter(
            'weights',
            nn.Parameter(torch.ones(self.n).float().unsqueeze(1), requires_grad=True)
        )

        self.to(self.device)

    def __forward__(self, model_output: ModelOutput):
        n_hidden_states = model_output.hidden_states[-self.n:]
        # stacked_cls = torch.cat([t.mean(dim=1) for t in n_hidden_states], 0).reshape(-1, self.n, 768)
        stacked_cls = torch.cat([t[:, 0, :] for t in n_hidden_states], 0).reshape(-1, self.n, 768)
        # return (stacked_cls * self.weights).mean(dim=1)
        return (stacked_cls * self.weights).reshape(stacked_cls.size()[0], -1)

    def __call__(self, *args, **kwargs):
        return self.__forward__(*args, **kwargs)


class ClusterLM(nn.Module):

    def __init__(self,
                 initial_centroids: torch.tensor,
                 lm_model,
                 tokenizer,
                 metric=lp_distance,
                 embedding_extractor=cls_embedding_extractor,
                 do_language_modeling=True,
                 device='cpu'
                 ):
        super(ClusterLM, self).__init__()

        self.initial_centroids = initial_centroids

        self.add_module('lm_model', lm_model)
        self.register_parameter('centroids', nn.Parameter(initial_centroids.clone().float(), requires_grad=True))

        self.tokenizer = tokenizer
        self.metric = metric
        self.embedding_extractor = embedding_extractor
        self.do_language_modeling = do_language_modeling
        self.device = device

        self.to(self.device)

    def forward(self, texts, alpha=1.0):
        """
        Input: texts and labels (optional)
        Returns: lm_language modelling output, own output dict (clustering_loss, predicted_labels)
        """
        # Language Modeling Part:

        lm_outputs = ModelOutput(loss=torch.tensor(0.0, requires_grad=True).to(self.device))

        if self.do_language_modeling:
            inputs = self.tokenizer(
                texts,
                return_tensors='pt',
                padding=True,
                truncation=True)

            input_ids = inputs['input_ids'].clone()
            input_ids, true_ids = mask_tokens(input_ids, self.tokenizer)
            inputs['input_ids'] = input_ids

            inputs = inputs.to(self.device)
            true_ids = true_ids.to(self.device)
            lm_outputs = self.lm_model(labels=true_ids, **inputs)

        # Clustering Part:
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True)

        inputs.to(self.device)

        # 0. Obtain embeddings for each input
        input_embeddings = self.embedding_extractor(self.lm_model.base_model(**inputs))

        # 1. Compute distances from each input embedding to each centroids
        distances = torch.stack([self.metric(embedding.unsqueeze(0), self.centroids) for embedding in input_embeddings])
        nearest_centroids = torch.argmin(distances.cpu().clone().detach(), dim=1)
        distances = torch.transpose(distances, 0, 1)  # => shape (n_centroids, n_samples)

        # 2. Compute the paramterized softmin for each centroid of each distance to each centroid per input sample
        # Find min distances for each centroid
        min_distances = torch.min(distances, dim=1).values
        # Compute exponetials
        exponentials = torch.exp(- alpha * (distances - min_distances.unsqueeze(1)))
        # Compute softmin
        softmin = exponentials / torch.sum(exponentials, dim=1).unsqueeze(1)

        # 3. Weight the distance between each sample and each centroid
        weighted_distances = distances * softmin

        # 4. Sum over weighted_distances to obtain loss
        clustering_loss = weighted_distances.sum(dim=1).mean()

        # Create clustering output dictionary
        cluster_outputs = ClusterOutput(
            loss=clustering_loss,
            predicted_labels=nearest_centroids.long(),
            embeddings=input_embeddings.cpu().detach()
        )

        return lm_outputs, cluster_outputs


def init_model(
        lm_model,
        tokenizer,
        data_loader,
        n_clusters,
        embedding_extractor=concat_cls_n_hidden_states,
        device='cpu',
        random_state=np.random.RandomState(42),
        **kwargs,
):
    initial_embeddings = []
    labels = []
    for batch_texts, batch_labels in data_loader:
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True)
        inputs = inputs.to(device)
        outputs = lm_model.base_model(**inputs)
        extracted_embeddings = embedding_extractor(outputs).cpu().detach().numpy()
        initial_embeddings.append(extracted_embeddings)
        labels.extend(batch_labels.numpy().astype('int'))

    initial_embeddings = np.vstack(initial_embeddings)

    initial_centroids = _k_init(
        initial_embeddings,
        n_clusters=np.unique(labels).shape[0],
        x_squared_norms=row_norms(initial_embeddings, squared=True),
        random_state=np.random.RandomState(42)
    )

    model = ClusterLM(
        lm_model=lm_model,
        tokenizer=tokenizer,
        embedding_extractor=embedding_extractor,
        initial_centroids=torch.from_numpy(initial_centroids),
        device=device,
        **kwargs
    )

    return model, initial_centroids, initial_embeddings


def evaluate(model, eval_data_loader, verbose=True):
    model.eval()
    true_labels = []
    predicted_labels = []
    eval_data_it = tqdm(eval_data_loader, desc='Eval') if verbose else eval_data_loader
    for batch_texts, batch_labels in eval_data_it:
        true_labels.extend(batch_labels.numpy().astype('int'))
        _, cluster_outputs = model(texts=list(batch_texts))
        predicted_labels.extend(cluster_outputs.predicted_labels.numpy().astype('int'))

    return np.array(predicted_labels), np.array(true_labels)


@dataclass
class TrainHistory:
    clustering_losses: List[float]
    lm_losses: List[float]
    combined_losses: List[float]
    prediction_history: List[np.array]
    eval_hist: List[Dict[str, float]]


def train(
        n_epochs,
        model,
        optimizer,
        scheduler,
        annealing_alphas,
        train_data_loader,
        eval_data_loader=None,
        do_eval=True,
        clustering_loss_weight=0.5,
        metrics=(cluster_accuracy, adjusted_rand_score, normalized_mutual_info_score),
        verbose=True
):
    total_clustering_losses = []
    total_lm_losses = []
    total_combined_losses = []
    prediction_history = []
    eval_hist = []

    assert len(annealing_alphas) >= n_epochs
    for epoch, alpha in zip(range(n_epochs), annealing_alphas):
        model.train()
        train_data_it = tqdm(train_data_loader, desc='Train') if verbose else train_data_loader
        for batch_texts, _ in train_data_it:
            lm_outputs, cluster_outputs = model(texts=list(batch_texts), alpha=alpha)
            combined_loss = lm_outputs.loss + (clustering_loss_weight * cluster_outputs.loss)

            optimizer.zero_grad()
            combined_loss.backward()
            optimizer.step()
            scheduler.step()

            total_clustering_losses.append(cluster_outputs.loss.item())
            total_lm_losses.append(lm_outputs.loss.item())
            total_combined_losses.append(combined_loss.item())

            if verbose:
                train_data_it.set_description(
                    f'Epoch: {epoch} | CombLoss: {combined_loss.item()} |LMLoss: {lm_outputs.loss.item()} | '
                    f' ClusterLoss: {cluster_outputs.loss.item()} | LR: {scheduler.get_last_lr()[0]} | Alpha: {alpha}'
                )

        if do_eval:
            if eval_data_loader is None:
                eval_data_loader = train_data_it if not verbose else train_data_it.iterable

            predicted_labels, true_labels = evaluate(
                model=model,
                eval_data_loader=eval_data_loader,
                verbose=verbose)

            prediction_history.append(deepcopy(predicted_labels))

            measurement = {}
            for metric in metrics:
                value = metric(true_labels, predicted_labels)
                measurement[metric.__name__] = value
                print(f'{metric.__name__}: {value}')
            eval_hist.append(measurement)

    train_history = TrainHistory(
        clustering_losses=total_clustering_losses,
        lm_losses=total_lm_losses,
        combined_losses=total_combined_losses,
        prediction_history=prediction_history,
        eval_hist=eval_hist
    )
    return train_history