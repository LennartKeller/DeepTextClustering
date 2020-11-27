from sacred import Experiment
import numpy as np
from tqdm import tqdm

ex = Experiment('ag_news_subset')

@ex.config
def cfg():

    n_epochs = 9
    lr = 2e-7
    lm_loss_weight = 1.0
    clustering_loss_weight = 0.025
    annealing_alpha = [1.5] * n_epochs
    random_state = np.random.RandomState(42)


@ex.automain
def run(n_epochs, lr, lm_loss_weight, clustering_loss_weight, annealing_alpha):
    pass


