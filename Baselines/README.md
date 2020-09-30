## Baselines

<newline>text<newline>
<div class="bg-yellow mb-2">
  .text-gray-dark on .bg-yellow
</div>


<a class="link-gray-dark"  href="#url">link-gray-dark</a>

#### Baselines Implemented
`1. doc2vec.py - Implements the Doc2vec model and classifies using Logistic Regression and Random Forests
2. H-LSTM.py  - Implements the History Based LSTM model, which extracts the content and meta features from the revisions (History) of each page and classifies using LSTM.`


Use `git status` to list all new or modified files that haven't yet been committed.
#### doc2vec.py

<newline> usage: doc2vec.py [-h] [--dataset_path [DATASET_PATH]] [--num_epoch [NUM_EPOCH]]

Read Arguments for doc2vec model

optional arguments:
  -h, --help            show this help message and exit
  --dataset_path [DATASET_PATH]
                        dataset path
  --num_epoch [NUM_EPOCH]
                        Number of epochs for doc2vec model  <newline>
