# NwQM
Implementation of the paper NwQM: A neural quality assessment framework for Wikipedia

#### Organization of the folder
```
1. Baselines - The Baseline models described in the paper are implemented here.
2. Data Processing - This folder contains code the generation of images (screenshots) of wikipedia pages from its URLs and also the text crawler for cleaning the text which is used in HAN etc.
3. Generate Embeddings - This folder contains the codes for generating the Finetuned BERT Embeddings for text pages, Google USE Embeddings for talk pages, Finetuned InceptionV3 Embeddings for Images.
4. NwQM Codes - The different NwQM Models such as NwQM, NwQM-w/oI (without Images), NwQM-w/oT (without Talk pages), NwQM-w/oTI (without Talk Pages and Images) are implemented here.
```
