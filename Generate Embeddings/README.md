This sections contains the codes for generating the Text page Embeddings, Talk page Embeddings and Image Embeddings used in the NwQM model.

#### Generate Embeddings 
```
1. Finetuned_BERT_Model.py                      - Finetunes the BERT model for Wikipedia's 6 class classification and saves the model.
2. Finetuned_InceptionV3_Model.py               - Finetunes the InceptionV3 model for Wikipedia's 6 class classification and saves the model.
3. Generate_Finetuned_BERT_Embeddings.py        - Generates the Finetuned BERT embeddings for each section of the text page from the Finetuned BERT model generated from Finetuned_BERT_Model.py and saves them.
4. Generate_Finetuned_InceptionV3_Embeddings.py - Generates the Finetuned InceptionV3 embeddings for the images (screenshot) of the wikipedia pages from the Finetuned InceptionV3 model generated from Finetuned_InceptionV3_Model.py and saves them.
5. Generate_Talkpage_Embeddings.py              - Generates the Talk page embeddings from Google Universal Sentence Encoder Model and saves them.
```

#### Finetuned_BERT_Model.py
To Run
```
usage: Finetuned_BERT_Model.py [-h] [--dataset_path [DATASET_PATH]] [--destination [DESTINATION]]
                               [--max_seq_length [MAX_SEQ_LENGTH]] [--num_finetune_layers [NUM_FINETUNE_LAYERS]]
                               [--num_epoch [NUM_EPOCH]] [--batch_size [BATCH_SIZE]] [--learning_rate [LEARNING_RATE]]

Read Arguments for Finetuned BERT model

optional arguments:
  -h, --help            show this help message and exit
  --dataset_path [DATASET_PATH]
                        dataset path
  --destination [DESTINATION]
                        Destination of saving Finetuned BERT Model
  --max_seq_length [MAX_SEQ_LENGTH]
                        Maximum number of tokens for each page/document
  --num_finetune_layers [NUM_FINETUNE_LAYERS]
                        Number of layers to be finetuned for BERT Model
  --num_epoch [NUM_EPOCH]
                        Number of epochs for Finetuned BERT model
  --batch_size [BATCH_SIZE]
                        Training batch size for Finetuned BERT model
  --learning_rate [LEARNING_RATE]
                        Learning rate for Finetuned BERT model
```
#### Finetuned_InceptionV3_Model.py
To Run
```
usage: Finetuned_InceptionV3_Model.py [-h] [--dataset_path [DATASET_PATH]] [--images_path [IMAGES_PATH]]
                                      [--destination [DESTINATION]] [--num_epoch [NUM_EPOCH]]
                                      [--batch_size [BATCH_SIZE]] [--learning_rate [LEARNING_RATE]]

Read Arguments for Finetuned InceptionV3 model

optional arguments:
  -h, --help            show this help message and exit
  --dataset_path [DATASET_PATH]
                        dataset path
  --images_path [IMAGES_PATH]
                        path of the folder containing images of the pages
  --destination [DESTINATION]
                        Destination of saving Finetuned InceptionV3 Model
  --num_epoch [NUM_EPOCH]
                        Number of epochs for Finetuned InceptionV3 model
  --batch_size [BATCH_SIZE]
                        Batch size of the Generator Object
  --learning_rate [LEARNING_RATE]
                        Learning rate for Finetuned InceptionV3 model
```                       
#### Generate_Finetuned_BERT_Embeddings.py
To Run
```
usage: Generate_Finetuned_BERT_Embeddings.py [-h] [--dataset_path [DATASET_PATH]] [--model_path [MODEL_PATH]]
                                             [--destination [DESTINATION]] [--max_seq_length [MAX_SEQ_LENGTH]]
                                             [--num_finetune_layers [NUM_FINETUNE_LAYERS]]
                                             [--learning_rate [LEARNING_RATE]]

Read Arguments for Finetuned BERT model

optional arguments:
  -h, --help            show this help message and exit
  --dataset_path [DATASET_PATH]
                        dataset path
  --model_path [MODEL_PATH]
                        Path of the Finetuned BERT Model
  --destination [DESTINATION]
                        Destination for saving Finetuned BERT Embeddings
  --max_seq_length [MAX_SEQ_LENGTH]
                        Maximum number of tokens for each page/document
  --num_finetune_layers [NUM_FINETUNE_LAYERS]
                        Number of layers to be finetuned for BERT Model
  --learning_rate [LEARNING_RATE]
                        Learning rate for Finetuned BERT model
```
#### Generate_Finetuned_InceptionV3_Embeddings.py
To Run
```
usage: Generate_Finetuned_InceptionV3_Embeddings.py [-h] [--dataset_path [DATASET_PATH]] [--images_path [IMAGES_PATH]]
                                                    [--model_path [MODEL_PATH]] [--destination [DESTINATION]]

Read Arguments for Finetuned InceptionV3 model

optional arguments:
  -h, --help            show this help message and exit
  --dataset_path [DATASET_PATH]
                        dataset path
  --images_path [IMAGES_PATH]
                        path of the folder containing images of the pages
  --model_path [MODEL_PATH]
                        Path of the Finetuned InceptionV3 Model
  --destination [DESTINATION]
                        Destination for saving Finetuned InceptionV3 Embeddings
``` 
#### Generate_Talkpage_Embeddings.py
To Run
```
usage: Generate_Talkpage_Embeddings.py [-h] [--dataset_path [DATASET_PATH]] [--destination [DESTINATION]]

Read Arguments for generating talk page embeddings

optional arguments:
  -h, --help            show this help message and exit
  --dataset_path [DATASET_PATH]
                        dataset path
  --destination [DESTINATION]
                        destination of generated talk page embeddings pckl file
```                        
