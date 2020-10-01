

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
To Run
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
#### H-LSTM.py
To Run
```
usage: H-LSTM.py [-h] [--dataset_path [DATASET_PATH]] [--files_path [FILES_PATH]] [--num_revisions [NUM_REVISIONS]]
                 [--only_cont] [--num_epoch [NUM_EPOCH]] [--batch_size [BATCH_SIZE]]

Read Arguments for History based LSTM model

optional arguments:
  -h, --help            show this help message and exit
  --dataset_path [DATASET_PATH]
                        dataset path
  --files_path [FILES_PATH]
                        wikipedia pages path
  --num_revisions [NUM_REVISIONS]
                        Number of revisions for each page
  --only_cont           If true use only content features else use both content and meta features
  --num_epoch [NUM_EPOCH]
                        Number of epochs for History based LSTM model
  --batch_size [BATCH_SIZE]
                        Training batch size for History based LSTM model
```  
#### HAN.py
To Run
```
usage: HAN.py [-h] [--dataset_path [DATASET_PATH]] [--glove_embed [GLOVE_EMBED]] [--num_epoch [NUM_EPOCH]]
              [--batch_size [BATCH_SIZE]]

Read Arguments for Hierarchical Attention Network model

optional arguments:
  -h, --help            show this help message and exit
  --dataset_path [DATASET_PATH]
                        dataset path
  --glove_embed [GLOVE_EMBED]
                        path of Glove Embeddings
  --num_epoch [NUM_EPOCH]
                        Number of epochs for HAN model
  --batch_size [BATCH_SIZE]
                        Training batch size for HAN model
```
#### HAN-wT.py
To Run
```
usage: HAN-wT.py [-h] [--dataset_path [DATASET_PATH]] [--glove_embed [GLOVE_EMBED]]
                 [--talk_embed_path [TALK_EMBED_PATH]] [--num_epoch [NUM_EPOCH]] [--batch_size [BATCH_SIZE]]

Read Arguments for Hierarchical Attention Network with talk model

optional arguments:
  -h, --help            show this help message and exit
  --dataset_path [DATASET_PATH]
                        dataset path
  --glove_embed [GLOVE_EMBED]
                        path of Glove Embeddings
  --talk_embed_path [TALK_EMBED_PATH]
                        path of generated talk page embeddings pckl file
  --num_epoch [NUM_EPOCH]
                        Number of epochs for HAN with talk model
  --batch_size [BATCH_SIZE]
                        Training batch size for HAN with talk model
```
#### DocBERT.py
To Run
```
usage: DocBERT.py [-h] [--dataset_path [DATASET_PATH]] [--max_seq_length [MAX_SEQ_LENGTH]]
                  [--num_finetune_layers [NUM_FINETUNE_LAYERS]] [--num_epoch [NUM_EPOCH]] [--batch_size [BATCH_SIZE]]
                  [--learning_rate [LEARNING_RATE]]

Read Arguments for DocBERT model

optional arguments:
  -h, --help            show this help message and exit
  --dataset_path [DATASET_PATH]
                        dataset path
  --max_seq_length [MAX_SEQ_LENGTH]
                        Maximum number of tokens for each page/document
  --num_finetune_layers [NUM_FINETUNE_LAYERS]
                        Number of layers to be finetuned for BERT Model
  --num_epoch [NUM_EPOCH]
                        Number of epochs for DocBERT model
  --batch_size [BATCH_SIZE]
                        Training batch size for DocBERT model
  --learning_rate [LEARNING_RATE]
                        Learning rate for DocBERT model
```                        
