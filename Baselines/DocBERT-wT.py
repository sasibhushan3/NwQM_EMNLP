''' This code implements the DocBERT with talk model
 
    We tokenize the cleaned text for each page (into BERT readable format) and pass the
    tokenized input to BERT model and finetune the BERT model.

    Every Wikipedia page has a talk page, So we use Google Universal Sentence Encoder 
    (Google USE) model to get the talk page representation.

    We then concatenate the page representation obtained from the finetuned BERT and
    talk page representation from Google USE and run a classification layer to classify
    into 6 Wikipedia classes.
                            (FA, GA, B, C, Start, Stub)
'''
# Tensorflow Version 1.x
import numpy as np
import pandas as pd 
import argparse
from tqdm import tqdm_notebook
import tensorflow as tf
import tensorflow_hub as hub
from tokenization import FullTokenizer
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate
from tensorflow import norm
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
    for example in tqdm_notebook(examples, desc="Converting examples to features"):
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


# Build the Final Model (DocBERT with Talk)
def build_model(max_seq_length, LearningRate, finetune_layers): 
    in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids")
    in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks")
    in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids")
    bert_output = BertLayer(n_fine_tune_layers=finetune_layers, pooling="first",
                            trainable = True)([in_id, in_mask, in_segment])
    text_rep  = tf.keras.layers.Dense(200, activation='relu')(bert_output)

    talk_inp = tf.keras.layers.Input(shape=(512,),name = 'talk_inp')
    talk_rep = tf.keras.layers.Dense(200, activation='relu')(talk_inp)
    norm1 = norm(text_rep-talk_rep,ord =2,keepdims = True,axis = -1)
    out1 = concatenate([text_rep,talk_rep,norm1],axis=-1)
    pred = tf.keras.layers.Dense(6, activation='softmax')(out1)
    optim = Adam(learning_rate=LearningRate, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model = tf.keras.models.Model(inputs=[in_id, in_mask, in_segment, talk_inp], outputs=pred)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
    return model


def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)


# Command line arguments
parser = argparse.ArgumentParser(description='Read Arguments for DocBERT with talk model')
parser.add_argument('--dataset_path', type=str, nargs='?', default='wikipages.csv',
                                        help='dataset path')
parser.add_argument('--talk_embed_path', type=str, nargs='?', default='talkpage_embeddings.pckl',
                                        help='path of generated talk page embeddings pckl file')
parser.add_argument('--max_seq_length', type=int, nargs='?', default=512,
                                        help='Maximum number of tokens for each page/document')
parser.add_argument('--num_finetune_layers', type=int, nargs='?', default=12,
                                        help='Number of layers to be finetuned for BERT Model')
parser.add_argument('--num_epoch', type=int, nargs='?', default=3,
                                        help='Number of epochs for DocBERT with talk model')
parser.add_argument('--batch_size', type=int, nargs='?', default=16,
                                        help='Training batch size for DocBERT with talk model')
parser.add_argument('--learning_rate', type=float, nargs='?', default=2e-5,
                                        help='Learning rate for DocBERT with talk model')
args = parser.parse_args()


# Params for bert model and tokenization
bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
max_seq_length = args.max_seq_length


# Read the Dataset
df = pd.read_csv(args.dataset_path)
df = df[0:30000]

pages_text = []
for k in range(30000):
    pages_text.append(df['Clean_Text'][k])


# Load the talk page representations generated from Google USE model
talk_data = np.zeros((len(df),512),dtype = 'float32')
with open(args.talk_embed_path, 'rb') as g:
    h = pk.load(g)
for i in range(len(df)):
    name = df['Name'][i]
    talk_data[i] = h[name]


# Convert the labels of each page in Numerical Format
labels = []
for i in range(len(df)):
    a = df['Label'][i]
    if(a == 'FA'):
        labels.append(0)
    if(a == 'GA'):
        labels.append(1)
    if(a == 'B'):
        labels.append(2)
    if(a == 'C'):
        labels.append(3)
    if(a == 'Start'):
        labels.append(4)
    if(a == 'Stub'):
        labels.append(5)


# Divide the data into train, val and test data
x_train = pages_text[:20000]
y_train = labels[:20000]
x_val = pages_text[20000:24000]
y_val = labels[20000:24000]
x_test = pages_text[24000:30000]
y_test = labels[24000:30000]
x_talk_train = talk_data[:20000]
x_talk_val = talk_data[20000:24000]
x_talk_test = talk_data[24000:30000]

train_text = np.array(x_train, dtype=object)[:, np.newaxis]
val_text = np.array(x_val, dtype=object)[:, np.newaxis]
test_text = np.array(x_test, dtype=object)[:, np.newaxis]
train_label = y_train
val_label = y_val
test_label = y_test


# Instantiate tokenizer
tokenizer = create_tokenizer_from_hub_module()


# Convert data to InputExample format
train_examples = convert_text_to_examples(train_text, train_label)
val_examples = convert_text_to_examples(val_text, val_label)
test_examples = convert_text_to_examples(test_text, test_label)


# Convert to features
(train_input_ids, train_input_masks, train_segment_ids, train_labels 
) = convert_examples_to_features(tokenizer, train_examples, max_seq_length=max_seq_length)
(val_input_ids, val_input_masks, val_segment_ids, val_labels
) = convert_examples_to_features(tokenizer, val_examples, max_seq_length=max_seq_length)
(test_input_ids, test_input_masks, test_segment_ids, test_labels
) = convert_examples_to_features(tokenizer, test_examples, max_seq_length=max_seq_length)


# Build DocBERT with talk model 
model = build_model(max_seq_length, args.learning_rate, args.num_finetune_layers)

# Instantiate variables
initialize_vars(sess)


# train the model
print("model fitting - DocBERT with talk Model")
model.fit([train_input_ids, train_input_masks, train_segment_ids, x_talk_train], train_labels,
          validation_data=([val_input_ids, val_input_masks, val_segment_ids, x_talk_val], val_labels),
          epochs=args.num_epoch, batch_size=args.batch_size)


# Predict and Evaluate the model
y_pred = model.predict([test_input_ids, test_input_masks, test_segment_ids, x_talk_train])
y_classes = y_pred.argmax(axis=-1)
evaluate = model.evaluate([test_input_ids, test_input_masks, test_segment_ids, x_talk_test],test_labels)

print("Accuracy obtained using DocBERT with talk model is : ",round(evaluate[1]*100,2))
print("Confusion Matrix of the results obtained using DocBERT with talk model is :")
print(confusion_matrix(test_labels,y_classes))


