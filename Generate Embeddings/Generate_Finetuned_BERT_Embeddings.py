''' This code generates the finetuned BERT embeddings for each section of a page from
    the BERT model which we finetuned.
'''
# Tensorflow Version 1.x
import numpy as np
import pandas as pd 
import pickle
import tensorflow as tf
import tensorflow_hub as hub
from tokenization import FullTokenizer
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix

# Initialize session
sess = tf.Session()

class PaddingInputExample(object):
"""Fake example so the num input examples is a multiple of the batch size.
   When running eval/predict on the TPU, we need to pad the number of examples
   to be a multiple of the batch size, because the TPU requires a fixed batch
   size. The alternative is to drop the last batch, which is bad because it means
   the entire output data won't be generated.
   We use this class instead of `None` because treating `None` as padding
   batches could cause silent errors.
"""


# A single training/test example for simple sequence classification.
class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
    """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'guid': self.guid,
            'text_a': self.text_a,
            'text_b': self.text_b,
            'label': self.label,
        })
        return config


# Implements the Bert Model and makes all the layers as trainable
class BertLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        n_fine_tune_layers=12,
        pooling="first",
        bert_path="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",
        **kwargs,
    ):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = True
        self.output_size = 768
        self.pooling = pooling
        self.bert_path = bert_path
        if self.pooling not in ["first", "mean"]:
            raise NameError(
                f"Undefined pooling type (must be either first or mean, but is {self.pooling}"
            )

        super(BertLayer, self).__init__(**kwargs)
        
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'n_fine_tune_layers': self.n_fine_tune_layers,
            'pooling': self.pooling,
            'bert_path': self.bert_path,
        })
        return config

    def build(self, input_shape):
        self.bert = hub.Module(self.bert_path, trainable=self.trainable, 
                               name=f"{self.name}_module")

        # Remove unused layers
        trainable_vars = self.bert.variables
        if self.pooling == "first":
            trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]
            trainable_layers = ["pooler/dense"]
        elif self.pooling == "mean":
            trainable_vars = [
                var
                for var in trainable_vars
                if not "/cls/" in var.name and not "/pooler/" in var.name
            ]
            trainable_layers = []
        else:
            raise NameError(
                f"Undefined pooling type (must be either first or mean, but is {self.pooling}"
            )

        # Select how many layers to fine tune
        for i in range(self.n_fine_tune_layers):
            trainable_layers.append(f"encoder/layer_{str(11 - i)}")

        # Update trainable vars to contain only the specified layers
        trainable_vars = [
            var
            for var in trainable_vars
            # if any([l in var.name for l in trainable_layers])
        ]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)

        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        if self.pooling == "first":
            pooled = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
                "pooled_output"
            ]
        elif self.pooling == "mean":
            result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
                "sequence_output"
            ]

            mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
            masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
                    tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)
            input_mask = tf.cast(input_mask, tf.float32)
            pooled = masked_reduce_mean(result, input_mask)
        else:
            raise NameError(f"Undefined pooling type (must be either first or mean, but is {self.pooling}")

        return pooled

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)



# Create the Bert Tokenizer
def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    bert_module =  hub.Module(bert_path)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    vocab_file, do_lower_case = sess.run(
        [
            tokenization_info["vocab_file"],
            tokenization_info["do_lower_case"],
        ]
    )

    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)



''' Converts a single `InputExample`(object for each page) into a single
    `InputFeatures`  (into the format readable by BERT model).
'''
def convert_single_example(tokenizer, example, max_seq_length=512):

    if isinstance(example, PaddingInputExample):
        input_ids = [0] * max_seq_length
        input_mask = [0] * max_seq_length
        segment_ids = [0] * max_seq_length
        label = 0
        return input_ids, input_mask, segment_ids, label

    tokens_a = tokenizer.tokenize(example.text_a)
    token_t = []
    len1 = len(tokens_a)

    ''' Consider only first 128 and last 382 tokens for each page,
        if the number of tokens exceed 510.
    '''
    if(len1 > 510):
        for k in range(128):
            token_t.append(tokens_a[k])
        for k in range(382):
            token_t.append(tokens_a[len1+k-382])
    else:
        for k in range(len1):
            token_t.append(tokens_a[k])
            
    if len(token_t) > max_seq_length - 2:
        tokens_a = token_t[0 : (max_seq_length - 2)]
    tokens_a = token_t

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids, example.label


# Convert a set of `InputExample`s to a list of `InputFeatures`.
def convert_examples_to_features(tokenizer, examples, max_seq_length=512):

    input_ids, input_masks, segment_ids, labels = [], [], [], []
    for example in examples:
        input_id, input_mask, segment_id, label = convert_single_example(
            tokenizer, example, max_seq_length
        )
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        labels.append(label)
    return (
        np.array(input_ids),
        np.array(input_masks),
        np.array(segment_ids),
        np.array(labels).reshape(-1, 1),
    )


# Create InputExamples object for each page
def convert_text_to_examples(texts, labels):
    
    InputExamples = []
    for text, label in zip(texts, labels):
        InputExamples.append(
            InputExample(guid=None, text_a=" ".join(text), text_b=None, label=label)
        )
    return InputExamples


# Build the Final Model
def build_model(max_seq_length, LearningRate, finetune_layers): 
    in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids")
    in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks")
    in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids")
    bert_inputs = [in_id, in_mask, in_segment]  
    bert_output = BertLayer(n_fine_tune_layers=finetune_layers,
                            pooling="first",trainable = True)(bert_inputs)
    pred = tf.keras.layers.Dense(6, activation='sigmoid')(bert_output)
    optim = Adam(learning_rate=LearningRate, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
    return model


def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)


# Command line arguments
parser = argparse.ArgumentParser(description='Read Arguments for Finetuned BERT model')
parser.add_argument('--dataset_path', type=str, nargs='?', default='wikipages.csv',
                                        help='dataset path')
parser.add_argument('--model_path', type=str, nargs='?', default='finetuned_bert_model.h5',
                                        help='Path of the Finetuned BERT Model')
parser.add_argument('--destination', type=str, nargs='?', default='finetuned_bert_embeddings.pckl',
                                        help='Destination for saving Finetuned BERT Embeddings')
parser.add_argument('--max_seq_length', type=int, nargs='?', default=512,
                                        help='Maximum number of tokens for each page/document')
parser.add_argument('--num_finetune_layers', type=int, nargs='?', default=12,
                                        help='Number of layers to be finetuned for BERT Model')
parser.add_argument('--learning_rate', type=float, nargs='?', default=2e-5,
                                        help='Learning rate for Finetuned BERT model')
args = parser.parse_args()


# Params for bert model and tokenization
bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
max_seq_length = args.max_seq_length


# Read the Dataset
df = pd.read_csv(args.dataset_path)

pages = []
for k in range(len(df)):
    pages.append(df['Clean_Text'][k])


# Convert each page into a list of sections
sec_list = []
for j in range(len(pages)):
    list1 = pages[j].split('Head_Token')
    page_sections = []
    for i in range(len(list1)):
        if(i==0):
            if(list1[i][-6:] == 'Level2'):
                page_sections.append(list1[i][:-6])
            else:
                page_sections.append(list1[i])
        elif(i>0):
            if(list1[i-1][-6:] == 'Level2'):
                str1 = 'Level2Head_Token'+list1[i]
                if(list1[i][-6:] == 'Level2'):
                    page_sections.append(str1[:-6])
                else:
                    page_sections.append(str1)
            else:
                str1 = 'Head_Token'+list1[i]
                if(list1[i][-6:] == 'Level2'):
                    page_sections.append(str1[:-6])
                else:
                    page_sections.append(str1)
    sec_list.append(page_sections)


# Pre-Process each section (sections which have more than 512 tokens)
sec_list_processed = []
for i in sec_list:
    list1 = []
    for j in i:
        temp = j.split()
        if(len(temp) < max_seq_length):
            t2 = ' '.join(temp)
        else:
            t2 = ' '.join(temp[0:max_seq_length])
        list1.append(t2)
    sec_list_processed.append(tn)


'''Combine the sections of all the pages into one single list,
   keeping track of number of sections for each page. 
'''
all_sections = []
num_sections = []
for i in range(len(df)):
    num_sections.append(len(sec_list_processed[i]))
    for j in range(len(sec_list_processed[i])):
        all_sections.append(sec_list_processed[i][j])


# Instantiate tokenizer
tokenizer = create_tokenizer_from_hub_module()


# Load the saved Finetuned BERT model
model = None
model = build_model(max_seq_length, args.learning_rate, args.num_finetune_layers)
initialize_vars(sess)
model.load_weights('/content/drive/My Drive/ftune_bert_3epoch.h5')


''' We need the fintuned BERT embedding from the model, So we need the intermediate
    layer output of the Finetuned model as the final layer of the model is the 
    classification layer.
'''
getlayer_output = K.function([model.layers[0].input,model.layers[1].input,model.layers[2].input],
                             [model.layers[3].output])


MAX_SECS = 16   # Pre-defined Parameter of maximum number of sections in a page 
EMBED_DIM = 768 # Dimension of BERT embedding


# Save all the BERT embeddings for each of a page in an array
data = np.zeros((len(df),MAX_SECS,EMBED_DIM), dtype='float32')


''' Calculate the BERT embeddings for each section, and add them to the
    corresponding page index of data array based on the number of sections
    in each page. 
'''
var = 0
for i in range(len(df)):
    if(num_sections[i] < MAX_SECS):
        #  Number of sections in a page are less than MAX_SECS
        cv = num_sections[i]
        diff = 0
    else:
        ''' If Number of sections in a page are greater than 
            MAX_SECS then ignore the sections after MAX_SECS.
        '''
        cv = MAX_SECS
        diff = num_sections[i] - MAX_SECS
    for j in range(cv):
        texttt = np.array([all_sections[var]], dtype=object)[:, np.newaxis]
        # 0 is just a temporary label, which has nothing to do with the model
        example = convert_text_to_examples(texttt, [0])
        (input_id, input_mask, 
         segment_id, l_label) = convert_examples_to_features(tokenizer, example,
                                                             max_seq_length=max_seq_length)
        post_save_preds = getlayer_output([input_id[0:], input_mask[0:], 
                                          segment_id[0:]])
        data[i][j] = post_save_preds[0][0]
        var+=1
    var +=diff

BERT_embed_dict = {}
for i in range(len(df)):
    name = df['Name'][i]
    BERT_embed_dict[name] = data[i]



# Save the computed BERT embeddings in a pickle file
with open(args.destination,'wb') as h:
    pickle.dump(BERT_embed_dict, h)
