@import '@primer/css/utilities/index.scss';
@import 'primer-buttons/index.scss';
// Import color variables for custom code
@import 'primer-support/index.scss';
// Override default blue
$blue: #0000ff;
@import './custom-that-uses-primer-variables.scss';
.foo {
  background: $blue;
  font-size: $h2-size;
  color: $text-gray;
}
## Baselines

<newline>text<newline>
<div class="bg-yellow mb-2">
  .text-gray-dark on .bg-yellow
</div>


<a class="link-gray-dark"  href="#url">link-gray-dark</a>

#### Baselines Implemented
`1. doc2vec.py - Implements the Doc2vec model and classifies using Logistic Regression and Random Forests <newline>
2. H-LSTM.py  - Implements the History Based LSTM model, which extracts the content and meta features from the revisions (History) of each page and classifies using LSTM.`



<div class="text-blue mb-2">
  .text-blue on white
</div>
<div class="text-gray-dark mb-2">
  .text-gray-dark on white
</div>
<div class="text-gray mb-2">
  .text-gray on white
</div>
<div class="text-red mb-2">
  .text-red on white
</div>
<div class="text-orange mb-2">
  .text-orange on white
</div>
<span class="float-left text-red tooltipped tooltipped-n" aria-label="Does not meet accessibility standards"><%= octicon("alert") %></span>
<div class="text-orange-light mb-2">
  .text-orange-light on white
</div>
<span class="float-left text-red tooltipped tooltipped-n" aria-label="Does not meet accessibility standards"><%= octicon("alert") %></span>
<div class="text-green mb-2 ml-4">
  .text-green on white
</div>
<div class="text-purple mb-2">
  .text-purple on white
</div>
  
  
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
