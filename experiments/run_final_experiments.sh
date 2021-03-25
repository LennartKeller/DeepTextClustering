#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate clustering

export CUDA_VISIBLE_DEVICES="1,2,3,4"
export MONGO_SACRED_ENABLED=false

python ag_news_subset5-distilbert.py ||
python ag_news_subset5-bert-base.py ||
python trec6-distilbert.py ||
python trec6-bert-base.py ||
python trec6-distilbert.py ||
python 20_newsgroups-distilbert.py ||
python 20_newsgroups-bert-base.py ||
