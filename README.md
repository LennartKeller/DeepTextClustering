# Deep text clustering with language models.

This repo contains the code and data for my term paper
"Fine-tuning language models on text clustering."

# Project structure

### ``transformers_clustering`` 
This folder contains the clustering module.
It should be installed using pip.

### ``experiments``
Contains the code for all experiments presented in the term paper
(and artefacts of old experiments).

### ``datasets``
Stores the datasets used for all experiments.

### ``results``
Will contain the results of each experiments.

### ``misc``
Contains some notebooks used to create plots and tables and some relicts of the development process.


# Dependencies:
Install all dependencies, either using pip or conda.

```
pip install -r requirements.txt
```  

or 

```
# This will create a new conda environment.
# Either:
conda create --name <env> --file requirements_conda.txt

# Or (better):
conda env create -f conda_environment.yml
# The latter will create a conda env named clustering, containing all required packages.
# Note: Depending on your system, you might have to change version of cuda-toolkit.
```

# Installation

Install the clustering module using pip:

```
# From project root.
pip install - e .
```

# Create datasets

Each datasets come with a script that downloads and creates the dataset as csv files.

### AG_News

To create the AG_News datasets 
first run the ``make_ag_news_datasets.py`` from the ``datasets/ag_news`` folder.
To create each subset and save the train and validation splits run the ``ag_news_create_splits.py``
from its location inside the ``datasets`` folder.`

### Trec 6 and 20 Newsgroups

To create these datasets just run their creation script from inside their folders.

# Run the experiments

The experiments have to be started from inside the ``experiments` folder.
Sacred is used to keep track of all experiments. 
If you want to enable sacred to save all results using MongoDB, you can specify
the following environments variables:

* MONGO_SACRED_ENABLED = true 
    * (If it is set to false the experiments will only be tracked using local files)
* mongo_user = MongoDB username
* mongo_pass = Password for the user
* mongo_host = Host of the MongoDB instance
* mongo_port = Port of the MongoDB instance

To run certain experiments without MongoDB tracking, you start it with the flag:

````
MONGO_SACRED_ENABLED=false python <experiment>.py
````

No matter if MongoDB is enabled each experiment will be tracked locally and the results are written 
to ``results/sacred_runs``
Parameters of each run can be changed using the sacred syntax:

````
python <experiment>.py with lr=1.0 n_epochs=1
```

The results of each run will be written to a 
folder following the pattern:  ``<project_root>/results/<experiment_name>/<timestamp>``


