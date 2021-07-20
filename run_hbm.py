#!/usr/bin/env python
# coding: utf-8

# ## Part 1 Build on word-level text to generate a fixed-length vector for each sentence



import numpy as np 
np.random.seed(2019)
from numpy import genfromtxt

import random as r
r.seed(2019)

import sys
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score,accuracy_score
from sklearn.metrics import confusion_matrix

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
torch.manual_seed(2019)
torch.cuda.manual_seed_all(2019)

import os
os.environ['PYTHONHASHSEED'] = str(2019)



from util import d, here

import pandas as pd
from argparse import ArgumentParser

import random, sys, math, gzip
from tqdm import tqdm
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

torch.backends.cudnn.benchmark = False

torch.backends.cudnn.deterministic = True

import gc
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import ShuffleSplit,StratifiedKFold
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import metrics
from sklearn.utils import shuffle
import glob
from optparse import OptionParser
import pickle




def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)




class Normalize(object):
    
    def normalize(self, X_train, X_val,max_len):
        self.scaler = MinMaxScaler()
        X_train, X_val = X_train.reshape(X_train.shape[0],-1),X_val.reshape(X_val.shape[0],-1)
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        
        X_train, X_val = X_train.reshape(X_train.shape[0],max_len,-1), X_val.reshape(X_val.shape[0],max_len,-1)
       
        return (X_train, X_val) 


    
    def inverse(self, X_train, X_val):
        X_train = self.scaler.inverse_transform(X_train)
        X_val   = self.scaler.inverse_transform(X_val)
    
        return (X_train, X_val) 


# ## Part 1. Import sentences embeddings


def import_data(dataset, max_len):



    with open('./datasets/roberta-base_data/%s_neg.p'%(dataset), 'rb') as fp:
        pre_trained_neg_dict = pickle.load(fp)
        
    with open('./datasets/roberta-base_data/%s_pos.p'%(dataset), 'rb') as fp:
        pre_trained_pos_dict = pickle.load(fp)



    pre_trained_neg = [pre_trained_neg_dict[i] for i in pre_trained_neg_dict.keys()]
    pre_trained_pos = [pre_trained_pos_dict[i] for i in pre_trained_pos_dict.keys()]



    pre_trained = pre_trained_neg+pre_trained_pos



    # Padding

    pos = np.zeros((len(pre_trained_pos),max_len,768)) 
    neg = np.zeros((len(pre_trained_neg),max_len,768))

    
    

    # assigning values
    for idx,doc in enumerate(pre_trained_pos): 
        if doc.shape[0]<=max_len:
    #         print(idx)
            pos[idx][:doc.shape[0],:] = doc
        else:
            pos[idx][:max_len,:] = doc[:max_len,:]
            
    print('positive example shape: ', pos.shape)
    
    # assert np.array_equal(pos[0],pre_trained_pos[0][:max_len,:])




    for idx,doc in enumerate(pre_trained_neg): 
        if doc.shape[0]<=max_len:
            neg[idx][:doc.shape[0],:] = doc
        else:
            neg[idx][:max_len,:] = doc[:max_len,:]
            
    print('negative example shape: ', neg.shape)
    print('negative example size', sizeof_fmt(sys.getsizeof(neg)))
    # assert np.array_equal(neg[0],pre_trained_neg[0][:max_len,:])




    TEXT_emb = np.concatenate((neg,pos),axis=0)
    assert TEXT_emb.shape == (len(pre_trained_neg)+len(pre_trained_pos),max_len,768)
    print(sizeof_fmt(sys.getsizeof(TEXT_emb)))



    del neg,pos

    LABEL_emb = np.array([0]*len(pre_trained_neg)+[1]*len(pre_trained_pos))

    return TEXT_emb, LABEL_emb


# ## Part 2. Config model




def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)


def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new, "mish": mish}





class BertConfig():
    """
        :class:`~transformers.BertConfig` is the configuration class to store the configuration of a
        `BertModel`.
        Arguments:
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu", "swish" and "gelu_new" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            seq_length: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
            layer_norm_eps: The epsilon used by LayerNorm.
    """


    def __init__(self,
                 hidden_size=768,
                 num_hidden_layers=4,
                 num_attention_heads=2,
                 intermediate_size=3072,
                 hidden_act="relu",
                 hidden_dropout_prob=0.01,
                 attention_probs_dropout_prob=0.01,
                 seq_length=128,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 output_attentions=True,
                 output_hidden_states=False,
                 num_labels=2):

  
  

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.seq_length = seq_length
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.num_labels = num_labels




# ## Part 3. Customized Sentence-level Transformer



BertLayerNorm = torch.nn.LayerNorm

class BertEmbeddings(nn.Module):
    """input sentence embeddings inferred by bottom pre-trained BERT, contruct position embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        
        self.config = config
        self.position_embeddings = nn.Embedding(config.seq_length, config.hidden_size)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, inputs_embeds, position_ids=None):
    
        input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        device = inputs_embeds.device
        
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)


        position_embeddings = self.position_embeddings(position_ids)
        
        embeddings = inputs_embeds + position_embeddings 
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertSelfAttention(nn.Module):
    
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        self.config = config
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size/config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertAttention(nn.Module):
    def __init__(self,config):
        super(BertAttention, self).__init__()
        self.config = config
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        

    def forward(self, hidden_states, attention_mask=None):
 
        self_outputs = self.self(hidden_states, attention_mask)
  
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        
        self.intermediate_act_fn = torch.nn.ReLU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.config = config
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.config = config
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        self_attention_outputs = self.attention(hidden_states, attention_mask)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights


        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs

class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.config = config
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask)
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
            
        
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)

class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply averaging hidden states
        mean_tensor = hidden_states.mean(dim=1)
        pooled_output = self.dense(mean_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertModel(nn.Module):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
        
        
    """
  
    
    
    def __init__(self, config):
        super(BertModel, self).__init__()
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
 

    def forward(self, inputs_embeds, attention_mask=None, position_ids=None):
        """ Forward pass on the Model.
        The model can behave as an encoder (with only self-attention) as well
        as a decoder, in which case a layer of cross-attention is added between
        the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
        Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    
        
        .. _`Attention is all you need`:
            https://arxiv.org/abs/1706.03762
        """
       
        input_shape = inputs_embeds.size()[:-1]

        

        device = inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)


        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError("Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(input_shape, attention_mask.shape))

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
   


        embedding_output = self.embeddings(inputs_embeds=inputs_embeds, position_ids=position_ids)
        encoder_outputs = self.encoder(embedding_output,
                                       attention_mask=extended_attention_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)

class HTransformer(nn.Module):
    """
    Sentence-level transformer, several transformer blokcs + softmax layer
    
    """

    def __init__(self, config):
        """
        :param emb_size: Embedding dimension
        :param heads: nr. of attention heads
        :param depth: Number of transformer blocks
        :param seq_length: Expected maximum sequence length
        :param num_classes: Number of classes.
     
        """
        super(HTransformer,self).__init__()
        
        self.config = config
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)


    def forward(self, x):
        """
        :param x: A batch by sequence length integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        outputs = self.bert(attention_mask=None,
                            position_ids=None,
                            inputs_embeds=x)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        
    
        outputs = (logits,) + outputs[2:]
        

        return outputs





def init_weights(module):
    """ Initialize the weights """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=config.initializer_range)
    elif isinstance(module, BertLayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()  



if __name__ == "__main__":

    # max_len = max(len(max(pre_trained_pos,key = lambda x: len(x))),len(max(pre_trained_neg,key = lambda x: len(x))))

    parser = OptionParser(usage='usage: %prog [training set size in list] dataset_name learning_rate no_epochs max_len')

    
    parser.add_option("-d","--dataset_name", action="store", type="string", dest="dataset_name", help="directory of data encoded by token-level Roberta", default = 'Longer_MovieReview')
    parser.add_option("-l","--learning_rate", action="store", type="float", dest="learning_rate", help="learning rate for fine tuning", default=2e-5)
    parser.add_option("-e","--no_epochs", action="store", type="int", dest="no_epochs", help="the number of epochs for fine tuning",default=50)
    parser.add_option("-m","--max_len", action="store", type="int", dest="max_len", help="the maximum sentences number per document",default=128)

    (options, args) = parser.parse_args()

    if len(args)<1:
        parser.error( "Must specify at least one number as training size and the largest training size is 200, you can customize the maximum training size by modifying the corrsponding codes of initializing training set." )

    try:
        for number in args:
             int(number)
    except: 
         parser.error( "Must specify at least one number as training size and the largest training size is 200, you can customize the maximum training size by modifying the corrsponding codes of initializing training set." )

    dataset = options.dataset_name
    max_len = options.max_len
    lr = options.learning_rate
    no_epochs = options.no_epochs


    gradient_clipping = 1.0
    train_batch=256
    val_batch=512

    # one can customize the maximum number instances in the training set by modifyting the corresponding codes of initializing training set
    train_sizes = [int(number) for number in args]
    

    cuda_num = 0

    # random states for subsampling training set
    seeds = [i for i in range(1988,1989)]

    TEXT_emb, LABEL_emb = import_data(dataset, max_len)
    config = BertConfig(seq_length=max_len)

    for train_size in train_sizes:
        for seed in seeds:
            
            confusion_matrices = []
            aucs = []
            

        
            ## initialize training set 

            index_shuffle = shuffle([i for i in range(len(TEXT_emb))], random_state=seed)
            
            total_train_shuffle = index_shuffle[:200]

            train_shuffle = total_train_shuffle[:train_size]
            test_shuffle = index_shuffle[200:]
            
            X_train,y_train = TEXT_emb[train_shuffle],LABEL_emb[train_shuffle]
            X_test, y_test = TEXT_emb[test_shuffle],LABEL_emb[test_shuffle]
            
            normalizer = Normalize()
            X_train, X_test = normalizer.normalize(X_train, X_test,max_len) 

            tensor_train_x = torch.from_numpy(X_train).type(torch.FloatTensor)
            tensor_train_y = torch.from_numpy(y_train).type(torch.LongTensor)

            tensor_val_x = torch.from_numpy(X_test).type(torch.FloatTensor)
            tensor_val_y = torch.from_numpy(y_test).type(torch.LongTensor)

            training_set = torch.utils.data.TensorDataset(tensor_train_x,tensor_train_y) # create your datset
            val_set = torch.utils.data.TensorDataset(tensor_val_x,tensor_val_y)

            trainloader=torch.utils.data.DataLoader(training_set, batch_size=train_batch, shuffle=True, num_workers=1)
            testloader=torch.utils.data.DataLoader(val_set, batch_size=val_batch, shuffle=False, num_workers=1)

           
            model = HTransformer(config=config)
            model.apply(init_weights)
            model.cuda(cuda_num)
            # model.to('cuda')
            opt = torch.optim.Adam(lr=lr, params=model.parameters())
            losses = []
            macro_f = []
            
            
            best_auc = 0.0
            best_acc = 0.0
            for e in tqdm(range(no_epochs)):
                
                

                print('\n epoch ',e)


                for i, data in enumerate(tqdm(trainloader)):

                    model.train(True)


                    opt.zero_grad()

                    inputs, labels = data

                    if inputs.size(1) > config.seq_length:
                        inputs = inputs[:, :config.seq_length, :]

                    if torch.cuda.is_available():
                        # inputs, labels = Variable(inputs.to('cuda')), labels.to('cuda')
                        inputs, labels = Variable(inputs.cuda(cuda_num)), labels.cuda(cuda_num)


                    out = model(inputs)


                    weight = [1.0,1.0]
                    print('balanced weight: ',weight)
                    weight = torch.tensor(weight).cuda(cuda_num)
                    # weight = torch.tensor(weight).to('cuda')
                    loss = nn.CrossEntropyLoss(weight,reduction='mean')

                    output = loss(out[0], labels)

                    print('epoch ',e,'step ',i,'loss:',output.item(),'num of postives', labels.sum())
                    train_loss_tol = float(output.cpu())

                    output.backward()


                    if gradient_clipping > 0.0:
                        nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

                    opt.step()



                    del inputs, labels, out, output
                    torch.cuda.empty_cache()


                    losses.append(train_loss_tol)          
            #         torch.save(model.state_dict(),'./Feasibility/%s_%s_%s'%(seed,e,i))


                    with torch.no_grad():
                        model.train(False)
                        y_eval_pred = []
                        y_eval_true = []
                        y_eval_prob = []
                        attention_scores = torch.Tensor()

                        for idx, data in enumerate(tqdm(testloader)):

                            inputs, labels = data


                            if inputs.size(1) > config.seq_length:
                                inputs = inputs[:, :config.seq_length, :]


                            if torch.cuda.is_available():
                                # inputs, labels = Variable(inputs.to('cuda')), labels.to('cuda')
                                inputs, labels = Variable(inputs.cuda(cuda_num)), labels.cuda(cuda_num)



                            out = model(inputs)
                            
                            sm = nn.Softmax(dim=1)
                            pred_prob = out[0].cpu()
                            pred_prob = sm(pred_prob)
                            
                            if config.output_attentions:
                                last_layer_attention = out[1][-1].cpu()
                                attention_scores = torch.cat((attention_scores, last_layer_attention))
        #                         attention_scores=attention_scores+[last_layer_attention]
                                

                            predict = torch.argmax(pred_prob, axis=1)
                            labels = labels.cpu()
                            y_eval_pred = y_eval_pred+predict.tolist()
                            y_eval_true = y_eval_true+labels.tolist()
                            y_eval_prob = y_eval_prob+pred_prob.tolist()
                            
                            y_eval_prob_pos = np.array(y_eval_prob)[:,1]

                            del inputs, labels, out
                        
                        

                      
                        
                        acc = accuracy_score(y_eval_true, y_eval_pred)
                        f_score = f1_score(y_eval_true,y_eval_pred,average='macro')
                        tn, fp, fn, tp = confusion_matrix(y_eval_true, y_eval_pred, labels=[0,1]).ravel()
                        
                        fpr, tpr, thresholds = metrics.roc_curve(y_eval_true, y_eval_prob_pos, pos_label=1)
                        step_auc = metrics.auc(fpr, tpr)
                        aucs.append(step_auc)
                        
                        confusion_matrices.append({'TP':tp, 'TN':tn, 'FP': fp, 'FN':fn})
                        


                        print(classification_report(y_eval_true,y_eval_pred),acc,f_score,metrics.auc(fpr, tpr))
                        macro_f.append(f_score)


                        print('best accuracy so far', best_acc)
                        print('accuracy this step', acc)

                        print('best auc so far', best_auc)
                        print('auc this step', step_auc)

                        # if best_auc<=step_auc:
                        #     best_auc = step_auc
                        #     files = glob.glob('./Feasibility/%s/attention/%s/*_%s.pt'%(dataset,seed,train_size))
                        #     for f in files:
                        #         os.remove(f)
                        #     doc_attention_score = []
                        #     for batch in attention_scores:
                        #         for doc in batch:
                        #             doc_attention_score.append(doc[0])
                                    
                            # final_attention_scores = np.zeros((len(TEXT_emb),config.seq_length))
                            # for idx,attention_matrix in enumerate(doc_attention_score):
                            #     attention_matrix = np.array(attention_matrix.cpu())
                            #     sum_attention_score = attention_matrix.sum(axis=0)
                            #     final_attention_scores[idx,:] = sum_attention_score
                                
                            # print(final_attention_scores.shape)
                            # torch.save(attention_scores,'./Feasibility/%s/attention/%s/%s_%s_%s.pt'%(dataset,seed,seed,e,train_size))

        #             fig, ax = plt.subplots(figsize=(12.5,10))
        #             plt.plot(losses,label='loss')
        #             plt.plot(macro_f,label='macro_f1')
        #             ax.legend()
                    
        #             plt.savefig('./Feasibility/%s_%s_%s_100'%(seed,e,i))
                    df = pd.DataFrame()
                    df[seed] = [row for row in confusion_matrices]
                    df.to_csv('./outputs/raw_%s_h%s_s%s_e%s_size%s.csv'%(dataset,config.num_attention_heads,seed,no_epochs,train_size),index=False)

                    
                    df_auc = pd.DataFrame()
                    df_auc[seed] = [row for row in aucs]
                    df_auc.to_csv('./outputs/auc_%s_h%s_s%s_e%s_size%s.csv'%(dataset,config.num_attention_heads,seed,no_epochs,train_size),index=False)
                    
                    
                    df_loss = pd.DataFrame()
                    df_loss[seed] = [row for row in losses]
                    df_loss.to_csv('./outputs/loss_%s_h%s_s%s_e%s_size%s.csv'%(dataset,config.num_attention_heads,seed,no_epochs,train_size),index=False)
            del model






