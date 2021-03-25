#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate clustering

export CUDA_VISIBLE_DEVICES="0,1,2,4"
export MONGO_SACRED_ENABLED=false

python ag_news_subset5-distilbert.py with device='cuda:2'
python ag_news_subset5-bert-base.py with device='cuda:2'
python trec6-distilbert.py with device='cuda:2'
python trec6-bert-base.py with device='cuda:2'
python 20_newsgroups-distilbert.py with device='cuda:2'
python 20_newsgroups-bert-base.py with device='cuda:2'