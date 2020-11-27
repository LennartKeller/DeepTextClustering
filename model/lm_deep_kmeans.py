from torch import nn


class DeepKMeansWithLM(nn.Module):

    def __init__(self, initial_centroids, lm_model, tokenizer):
        super(DeepKMeansWithLM, self).__init__()
