This sections contains the codes for generating the Text page Embeddings, Talk page Embeddings and Image Embeddings used in the NwQM model.

#### Generate Embeddings 
```
1. Finetuned_BERT_Model.py                      - Finetunes the BERT model for Wikipedia's 6 class classification and saves the model.
2. Finetuned_InceptionV3_Model.py               - Finetunes the InceptionV3 model for Wikipedia's 6 class classification and saves the model.
3. Generate_Finetuned_BERT_Embeddings.py        - Generates the Finetuned BERT embeddings for each section of the text page from the Finetuned BERT model generated from Finetuned_BERT_Model.py and saves them.
4. Generate_Finetuned_InceptionV3_Embeddings.py - Generates the Finetuned InceptionV3 embeddings for the images (screenshot) of the wikipedia pages from the Finetuned InceptionV3 model generated from Finetuned_InceptionV3_Model.py and saves them.
5. Generate_Talkpage_Embeddings.py              - Generates the Talk page embeddings from Google Universal Sentence Encoder Model and saves them.
```
