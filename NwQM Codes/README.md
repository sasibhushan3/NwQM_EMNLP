

#### NwQM.py
To Run
```
usage: NwQM.py [-h] [--dataset_path [DATASET_PATH]] [--text_embed_path [TEXT_EMBED_PATH]]
               [--talk_embed_path [TALK_EMBED_PATH]] [--image_embed_path [IMAGE_EMBED_PATH]] [--num_epoch [NUM_EPOCH]]
               [--batch_size [BATCH_SIZE]] [--learning_rate [LEARNING_RATE]]

Read Arguments for NwQM model

optional arguments:
  -h, --help            show this help message and exit
  --dataset_path [DATASET_PATH]
                        dataset path
  --text_embed_path [TEXT_EMBED_PATH]
                        path of generated text embeddings pckl file
  --talk_embed_path [TALK_EMBED_PATH]
                        path of generated talk page embeddings pckl file
  --image_embed_path [IMAGE_EMBED_PATH]
                        path of generated image embeddings pckl file
  --num_epoch [NUM_EPOCH]
                        Number of epochs for NwQM model
  --batch_size [BATCH_SIZE]
                        Training batch size for NwQM model
  --learning_rate [LEARNING_RATE]
                        Learning rate for NwQM model
 ```                       
#### NwQM_wo_I.py
To Run
```
usage: NwQM_wo_I.py [-h] [--dataset_path [DATASET_PATH]] [--text_embed_path [TEXT_EMBED_PATH]]
                    [--talk_embed_path [TALK_EMBED_PATH]] [--num_epoch [NUM_EPOCH]] [--batch_size [BATCH_SIZE]]
                    [--learning_rate [LEARNING_RATE]]

Read Arguments for NwQM-w/oI model

optional arguments:
  -h, --help            show this help message and exit
  --dataset_path [DATASET_PATH]
                        dataset path
  --text_embed_path [TEXT_EMBED_PATH]
                        path of generated text embeddings pckl file
  --talk_embed_path [TALK_EMBED_PATH]
                        path of generated talk page embeddings pckl file
  --num_epoch [NUM_EPOCH]
                        Number of epochs for NwQM-w/oI model
  --batch_size [BATCH_SIZE]
                        Training batch size for NwQM-w/oI model
  --learning_rate [LEARNING_RATE]
                        Learning rate for NwQM-w/oI model
```
#### NwQM_wo_I.py
To Run
```
usage: NwQM_wo_T.py [-h] [--dataset_path [DATASET_PATH]] [--text_embed_path [TEXT_EMBED_PATH]]
                    [--image_embed_path [IMAGE_EMBED_PATH]] [--num_epoch [NUM_EPOCH]] [--batch_size [BATCH_SIZE]]
                    [--learning_rate [LEARNING_RATE]]

Read Arguments for NwQM-w/oT model

optional arguments:
  -h, --help            show this help message and exit
  --dataset_path [DATASET_PATH]
                        dataset path
  --text_embed_path [TEXT_EMBED_PATH]
                        path of generated text embeddings pckl file
  --image_embed_path [IMAGE_EMBED_PATH]
                        path of generated image embeddings pckl file
  --num_epoch [NUM_EPOCH]
                        Number of epochs for NwQM-w/oT model
  --batch_size [BATCH_SIZE]
                        Training batch size for NwQM-w/oT model
  --learning_rate [LEARNING_RATE]
                        Learning rate for NwQM-w/oT model
```
#### NwQM_wo_TI.py
To Run
```
usage: NwQM_wo_TI.py [-h] [--dataset_path [DATASET_PATH]] [--text_embed_path [TEXT_EMBED_PATH]]
                     [--num_epoch [NUM_EPOCH]] [--batch_size [BATCH_SIZE]] [--learning_rate [LEARNING_RATE]]

Read Arguments for NwQM-w/oTI model

optional arguments:
  -h, --help            show this help message and exit
  --dataset_path [DATASET_PATH]
                        dataset path
  --text_embed_path [TEXT_EMBED_PATH]
                        path of generated text embeddings pckl file
  --num_epoch [NUM_EPOCH]
                        Number of epochs for NwQM-w/oTI model
  --batch_size [BATCH_SIZE]
                        Training batch size for NwQM-w/oTI model
  --learning_rate [LEARNING_RATE]
                        Learning rate for NwQM-w/oTI model
```
