

#### Baselines Implemented
```
1. doc2vec.py     - Implements the Doc2vec model and classifies using Logistic Regression and Random Forests.
2. H-LSTM.py      - Implements the History Based LSTM model, which extracts the content and meta features from the revisions (History) of each page and classifies using LSTM.
3. HAN.py         - Implements the 4 level Hierarchical Attention Network (HAN) model for Wikipedia Pages.
4. HAN-wT.py      - Implements the 4 level Hierarchical Attention Network (HAN) model with talk page representations for Wikipedia Pages.
5. DocBERT.py     - Implements the DocBERT model where we tokenize each page (into BERT readable format) and pass the tokenized input to BERT model and finetune the BERT model for classification.
6. DocBERT-wT.py  - Implements the DocBERT with talk model where we add the talk page representation to the BERT representation and finetune the BERT model for classification.
7. InceptionV3.py - Implements the InceptionV3 model which uses the image (screenshot) of each page and is finetuned for classification.
8. M-BILSTM.py    - Implements the joint model where we use BILSTM to get the text page representation and Finetuned InceptionV3 model for the image representation and combined together for classification.
9. Talk.py        - Implement the classification task by considering only the talk pages for each page.
```

#### doc2vec.py
```
usage: doc2vec.py [-h] [--dataset_path [DATASET_PATH]] [--num_epoch [NUM_EPOCH]]

Read Arguments for doc2vec model

optional arguments:
  -h, --help            show this help message and exit
  --dataset_path [DATASET_PATH]
                        dataset path
  --num_epoch [NUM_EPOCH]
                        Number of epochs for doc2vec model
```
