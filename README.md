# ClusterLog: Unsupervised Clusterization of Error Messages

**Requirements:**
```
Python >= 3.7 < 3.8
```
This package doesn't work currently with python 2.7 because of `kneed` library, and with python 3.8 because of `gensim`.

```
editdistance==0.5.3
gensim==3.4.0
kneed==0.4.1
nltk==3.4.5
numpy==1.16.4
pandas==0.25.1
pyonmttok==1.10.1
scikit-learn==0.21.2
```

**Input:**
   Pandas DataFrame with error log messages. DataFrame may have arbitrary columns and column names, but
   it must contain index column with IDs and column with text log messages. The name of log column is not
   fixed, but it must be specified explicitly in settings as 'target'.
   Possible structure of DataFrame is the following (in this example, `tagret='log_message'`):
   ```
   ID   |   log_message                                                            | timestamp
   -----------------------------------------------------------------------------------------------------
    1   |   No events to process: 16000 (skipEvents) >= 2343 (inputEvents of HITS  | 2019-10-01T10:18:49
    2   |   AODtoDAOD got a SIGKILL signal (exit code 137)                         | 2019-10-01T09:01:57
    ...
   ```
Required input:
- `df`
- `target`

Optional input:

- clusterization_settings
    - `w2v_size` (default: 100)
    - `w2v_window` (default: 7)
    - `min_samples` (default: 1)
- `model_name` (path to a file with word2vec model)
- `mode` ('create'(default) | 'update' | 'load')

**Modes:**
1) `create`
    - Create word2vec model based on large sample of error logs
    - Save it to file ‘word2vec.model’ on server for further usage
2) `process`
    - Load word2vec model from file (without re-training the model)
3) `update`
    - Load word2vec model from file and train (update) this model with new error logs
    - Save updated model in file


**Clusterization of error log messages is implemented as a chain of methods:**

1) `data_preparation` - cleaning initial log messages from all substrings with digits
2) `grouping equals` - group dataframe by equal cleaned messages
2) `tokenization` - split each log message into tokens (`pyonmttok` + retaining spaces)
3) `tokens_vectorization` - train word2vec model
4) `sentence_vectorization` - convert word2vec to sent2vec model
5) `kneighbors` - calculate k-neighbors
6) `epsilon_search` - search epsilon for the DBSCAN algorithm
7) `dbscan` - execute DBSCAN clusterization, returns cluster labels
8) `reclusterization` - reclustering the existing clusters using the Levenshtein distances between sequences of tokens
9) `validation` - calculating similarity score for each cluster

**Output:**

The output is available in different views:
   1) `ALL` - DataFrame grouped by cluster numbers
   2) `INDEX` - dictionary of lists of indexes for all clusters
   3) `TARGET` - dictionary of lists of error messages for all clusters
   4) `cluster labels` - array of cluster labels (as output of `DBSCAN -> fit_predict()`)


**Clusters statistics:**

Clusters Statistics returns DataFrame or dictionary with statistic for all clusters:
- `cluster_name` - name of a cluster
- `cluster_size` - number of log messages in cluster
- `pattern` - all common substrings in messages in the cluster
- `mean_similarity` - average similarity of log messages in cluster
- `std_similarity` - standard deviation of similarity of log messages in cluster
- `indices` - indices of the initial dataframe, corresponding to the cluster


**Installation:**

```
pip install clusterlogs
```

**Usage:**
```
from clusterlogs import pipeline
```

Detailed usage of this library is described at
[clusterlogs_notebook.ipynb](https://github.com/maria-grigorieva/ClusterLog/blob/master/clusterlogs_notebook.ipynb).


**Author:**
maria.grigorieva@cern.ch (Maria Grigorieva)
