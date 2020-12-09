## Protocol

Adapted from ```Moradi Fard, M., Thonet, T., & Gaussier, E. (2020). Deep k-Means: Jointly clustering with k-Means and learning representations. Pattern Recognition Letters. doi:10.1016/j.patrec.2020.07.028```

## Number of clusters

* We only use datasets which provide class labels for each instance

* The number of clusters is determined by the number of classes

### Splits

* Each dataset is splitted randomly into a training-set (90%) and a validation-set(10%)

* The different subset of ag_news are treated as independent datasets

* For the 20newgroups dataset the splits provided by the Moradi et. al are used

#### Hyperparam search

* The traininig-set consists of the training- and validation-set

* The performance is only measured on the validation-set

* We set the maximum number of epochs to 5 and enable early stopping with a tolerance of 1% of label changes

__Ag News__:

* We train all hyperparams except the learning rate on ag_news_subset5 and use them for all ag_news-subsets

* The learning rate is tuned on each ag_news-subset
  
### Training and Evaluation

* Because clustering is a unsupervised learning task we train and evaluate on the training-set

* We set the maximum number of epochs to 10 and enable early stopping with a tolerance of 1% of label changes

### Metrics
To evaluate the performance of this model we use supervised measures (i.e. measure that score performance based on gold standard labels)

We use the following metrics:

* Cluster accuracy: First the best matching between class labels and cluster indices is computed and then the accuarcy is computed based on the matching
  * We use this measure to compute how well the classes can be found by clustering
* Adjusted-Rand-Index:
* Normalized Mutual Information Score
