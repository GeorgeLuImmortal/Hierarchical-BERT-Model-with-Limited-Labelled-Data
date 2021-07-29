

import os
os.environ['PYTHONHASHSEED'] = str(2019)

import re

import numpy as np
np.random.seed(2019)

import random as r
r.seed(2019)

from tensorflow import set_random_seed
set_random_seed(2019)

import pandas as pd
from bs4 import BeautifulSoup
from keras import backend as K
from keras.models import Model
from keras import initializers,callbacks
from keras.engine.topology import Layer
from keras.layers import Dense, Input
from keras.layers import Embedding, GRU, Bidirectional, TimeDistributed
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils.np_utils import to_categorical
from nltk import tokenize
from sklearn.utils import shuffle
from sklearn import metrics
from keras import optimizers
from nltk import tokenize
from optparse import OptionParser
from sklearn.metrics import confusion_matrix
import string

args = {
        'batch_size': 16,
        'maxlen' : 100,
        'max_sentences' : 100,
        'max_words' : 20000,
        'embedding_dim' : 200,
        'glove_dir' : "./",
        'embeddings_index' : {},
        'text_data_dir': 'raw_corpora/',
        'output_dir': 'outputs/'

}

 ## take a list of strings as optional arguments
def list_callback(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))


# class defining the custom attention layer
class HierarchicalAttentionNetwork(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(HierarchicalAttentionNetwork, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim,)))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(HierarchicalAttentionNetwork, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))

        ait = K.exp(K.squeeze(K.dot(uit, self.u), -1))

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        weighted_input = x * K.expand_dims(ait)
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


def remove_html(str_a):
    p = re.compile(r'<.*?>')
    return p.sub('', str_a)


# replace all non-ASCII (\x00-\x7F) characters with a space
def replace_non_ascii(str_a):
    return re.sub(r'[^\x00-\x7f]', r'', str_a)


## lowercase, remove digits, non-alphabetic chars, punctuations
## and extra spaces

def clean_corpus(corpus):
    
    cleaned_corpus = []
    
    for article in corpus:

        article = article.lower()
        temp_str = re.sub(r'\d+', '', article)
        temp_str = re.sub(r'[^\x00-\x7f]',r'', temp_str)
        temp_str = temp_str.translate(str.maketrans('', '', string.punctuation))
        temp_str = re.sub(r'\s+', ' ', temp_str)


        cleaned_corpus.append(temp_str)
        
    return cleaned_corpus


def import_data(dataset):


    df_neg_text = pd.read_csv(os.path.join(args['text_data_dir'],'%s/%s_neg_text.csv'%(dataset,dataset)))
    df_pos_text = pd.read_csv(os.path.join(args['text_data_dir'],'%s/%s_pos_text.csv'%(dataset,dataset)))


    # df_neg_text['text'] = clean_corpus(df_neg_text.text.tolist())
    # df_pos_text['text'] = clean_corpus(df_pos_text.text.tolist())

    ## raw text
    texts = df_neg_text.text.tolist()+df_pos_text.text.tolist()

    ## labels 
    labels = np.array([0]*len(df_neg_text)+[1]*len(df_pos_text))

    print(len(texts),np.bincount(labels))

    labels = to_categorical(np.asarray(labels))
    ## segmented documents
    reviews = []

    # for idx,document in enumerate(texts):
    #     temp_seg_doc = []
    #     for sentence in document.split('\n'):
    #         if len(sentence.split())>2:
    #             temp_seg_doc.append(sentence.strip())
    #     reviews.append(temp_seg_doc)
        
    for idx,document in enumerate(texts):
        temp_seg_doc = []
        for sentence in tokenize.sent_tokenize(document):
            if len(sentence.split())>2:
                temp_seg_doc.append(sentence.strip().lower())
        reviews.append(temp_seg_doc)
        

    tokenizer = Tokenizer(num_words=args['max_words'])
    tokenizer.fit_on_texts(texts)

    data = np.zeros((len(texts), args['max_sentences'], args['maxlen']), dtype='int32')

    for i, sentences in enumerate(reviews):
        for j, sent in enumerate(sentences):
            if j < args['max_sentences']:
                wordTokens = text_to_word_sequence(sent)
                k = 0
                for _, word in enumerate(wordTokens):
                    if k < args['maxlen'] and tokenizer.word_index[word] < args['max_words']:
                        data[i, j, k] = tokenizer.word_index[word]
                        k = k + 1
                        
    word_index = tokenizer.word_index
    print('Total %s unique tokens.' % len(word_index))
    print('Shape of reviews (data) tensor:', data.shape)
    print('Shape of sentiment (label) tensor:', labels.shape)


    return data, labels, df_neg_text, df_pos_text, word_index


if __name__ == "__main__":

    # max_len = max(len(max(pre_trained_pos,key = lambda x: len(x))),len(max(pre_trained_neg,key = lambda x: len(x))))

    parser = OptionParser(usage='usage: -r random_seeds -d dataset_name -l learning_rate -e no_epochs -s train_size')

    
    parser.add_option("-d","--dataset_name", action="store", type="string", dest="dataset_name", help="directory of data encoded by token-level Roberta", default = 'longer_moviereview')
    parser.add_option("-l","--learning_rate", action="store", type="float", dest="learning_rate", help="learning rate", default=1e-3)
    parser.add_option("-e","--no_epochs", action="store", type="int", dest="no_epochs", help="the number of epochs",default=50)
    parser.add_option('-r', '--random_seeds', type='string', action='callback',dest='random_seeds',callback=list_callback,default=['1988','1989'])
    parser.add_option('-s', '--training_size', type='string', action='callback',dest='training_size',callback=list_callback,default=['50','100'])

    (options, _) = parser.parse_args()


    for number in options.training_size:
        if int(number)>200:
            parser.error( "The largest training size is 200, you can customize the maximum training size by modifying the corrsponding codes of initializing training set." )

    dataset = options.dataset_name
    lr = options.learning_rate
    no_epochs = options.no_epochs
    embeddings_index = args['embeddings_index']

    # one can customize the maximum number instances in the training set by modifyting the corresponding codes of initializing training set
    train_sizes = [int(number) for number in options.training_size]
    random_states = [int(number) for number in options.random_seeds]

    print('number of epochs: ', no_epochs)
    print('dataset name: ', dataset)
    print('initial random states: ', random_states)
    print('training set sizes: ', train_sizes)


  
    data, labels,df_neg_text, df_pos_text, word_index = import_data(dataset)


    for idx,train_size in enumerate(train_sizes):
        
        df_all = pd.DataFrame()
        df_all_auc = pd.DataFrame()

        accs = []
        aucs = []
        confusion_matrices = []
        
        for seed in random_states:

            index_shuffle = shuffle([i for i in range(data.shape[0])], random_state=seed)

            total_train_shuffle = index_shuffle[:200]
            train_shuffle = total_train_shuffle[:train_size]
            test_shuffle = index_shuffle[200:]

            y_categorical = np.array([0]*len(df_neg_text)+[1]*len(df_pos_text))

            x_train,y_train = data[train_shuffle],labels[train_shuffle]
            x_val, y_val = x_train,y_train
            x_test,y_test,y_test_cate = data[test_shuffle], labels[test_shuffle],y_categorical[test_shuffle]

            print('Number of positive and negative reviews in training and validation set')
            print(y_train.sum(axis=0))
            print(y_val.sum(axis=0))

            ## prtrained GloVe embeddings downloaded from https://www.kaggle.com/incorpes/glove6b200d
            f = open(os.path.join(args['glove_dir'], 'glove.6B.200d.txt'),encoding='utf8')
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            f.close()

            print('Total %s word vectors.' % len(embeddings_index))

            # building Hierachical Attention network
            embedding_matrix = np.random.random((len(word_index) + 1, args['embedding_dim']))
            for word, i in word_index.items():
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    embedding_matrix[i] = embedding_vector

            embedding_layer = Embedding(len(word_index) + 1, args['embedding_dim'], weights=[embedding_matrix],
                                        input_length=args['maxlen'], trainable=True, mask_zero=True)

            sentence_input = Input(shape=(args['maxlen'],), dtype='int32')
            embedded_sequences = embedding_layer(sentence_input)
            lstm_word = Bidirectional(GRU(50, return_sequences=True))(embedded_sequences)
            attn_word = HierarchicalAttentionNetwork(100)(lstm_word)
            sentenceEncoder = Model(sentence_input, attn_word)

            review_input = Input(shape=(args['max_sentences'], args['maxlen']), dtype='int32')
            review_encoder = TimeDistributed(sentenceEncoder)(review_input)
            lstm_sentence = Bidirectional(GRU(100, return_sequences=True))(review_encoder)
            attn_sentence = HierarchicalAttentionNetwork(100)(lstm_sentence)
            preds = Dense(2, activation='softmax')(attn_sentence)
            model = Model(review_input, preds)
            
            callback = callbacks.EarlyStopping(monitor='loss', patience=3,min_delta=1e-4)
            opt = optimizers.Adam(lr=lr)
            model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])

            print("model fitting - Hierachical attention network")

            his = model.fit(x_train, y_train, validation_data=(x_val, y_val), nb_epoch=no_epochs, batch_size=args['batch_size'],callbacks=[callback])

            y_predict = model.predict(x_test)
            y_eval_prob_pos = np.array(y_predict)[:,1]

            y_pred = np.argmax(y_predict,axis=1)
            acc = metrics.accuracy_score(y_test_cate, y_pred)
            accs.append(acc)

            fpr, tpr, thresholds = metrics.roc_curve(y_test_cate, y_eval_prob_pos,pos_label=1)
            auc = metrics.auc(fpr, tpr)
            aucs.append(auc)
            
            tn, fp, fn, tp = confusion_matrix(y_test_cate, y_pred, labels=[0,1]).ravel()
            confusion_matrices.append({'TP':tp, 'TN':tn, 'FP': fp, 'FN':fn})
          


        df_all['result'] = [row for row in confusion_matrices]
        df_all['seed'] = random_states
        df_all = df_all.set_index('seed')
        df_all.to_csv(os.path.join(args['output_dir'],'raw_%s_han_%s.csv'%(dataset,train_size)),index=True)

        df_all_auc['result'] = [row for row in aucs]
        df_all_auc['seed'] = random_states
        df_all_auc = df_all_auc.set_index('seed')
        df_all_auc.to_csv(os.path.join(args['output_dir'],'auc_%s_han_%s.csv'%(dataset,train_size)),index=True)


