import torch
from transformers import *
torch.manual_seed(2019)
torch.cuda.manual_seed_all(2019)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import random as r
r.seed(2019)

import numpy as np
np.random.seed(2019)
import pandas as pd
from operator import add

import re
import string

from nltk import sent_tokenize
from tqdm import tqdm

import pickle
import matplotlib.pyplot as plt
from optparse import OptionParser

from gensim.models import fasttext

import os
os.environ['PYTHONHASHSEED'] = str(2019)

args = {
        'cuda_num': 1,

        ## directory for original text data
        'text_data_dir': 'corpus_data/',

        ## directory for encoded embeddings
        'data_dir': 'datasets/',

}


def list_callback(option, opt, value, parser):
    setattr(parser.values, option.dest, value.split(','))

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





def read_data(path):

    corpora = []
    for filename in os.listdir(path):

        df_temp = pd.read_csv(path+filename, encoding='iso-8859-1')

        corpora.append(df_temp.text.tolist())

    class_one_len = len(corpora[0])
    class_two_len = len(corpora[1])

    return corpora, class_one_len, class_two_len





## split a document into sentences
def sentences_segmentation(corpora,tokenizer,min_token=0):
    segmented_documents = []
    
    for document in tqdm(corpora):

        segmented_document = []
        seg_document = sent_tokenize(document)

        ## remove sentences that are too short, the tokenized sentences should larger than min_token, otherwise are dropped
        for sentence in seg_document:
            tokenized_sentence = tokenizer.tokenize(sentence)
            if len(tokenized_sentence)>min_token:
                temp_sentence = tokenizer.convert_tokens_to_string(tokenized_sentence)
                ## if a whole sentence consists of punctations, it will be dropped
                if not all([j.isdigit() or j in string.punctuation for j in temp_sentence]):
                    segmented_document.append(temp_sentence)
            
        segmented_documents.append(segmented_document)
    
    return segmented_documents




def encode_fasttext(dataset_name):

    corpora, class_one_len, class_two_len = read_data(os.path.join(args['text_data_dir'],'%s/'%(dataset_name)))

    ## pretrained model downloaded from https://fasttext.cc/docs/en/pretrained-vectors.html
    model = fasttext.load_facebook_model('wiki.en.bin')
    
    all_corpus = corpora[0]+corpora[1]
    cleaned_corpus = clean_corpus(all_corpus)

    print('total number of examples ',len(cleaned_corpus),'\n')
    
    representations=[]
    
    for article in cleaned_corpus:
        tokenized_article = article.split()
        corpus = [word for word in tokenized_article if word != '']
        matrix = []
    #     print(corpus)
        for word in corpus:
            matrix.append(model.wv[word])
        matrix = np.array(matrix)
    #     print(matrix.shape)
        representations.append(np.mean(matrix,0))
        
    representations = np.array(representations)

    print('represetation shape', representations.shape) 
    
    return representations,class_one_len,class_two_len


# for roberta and roberta_hbm, we suggest do not apply any preprocessing
def encode_roberta(dataset_name):
    
   # Load pre-trained model tokenizer (vocabulary)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # Load pre-trained model (weights)
    model = RobertaModel.from_pretrained('roberta-base')

    # Set the model in evaluation mode to desactivate the DropOut modules
    # This is IMPORTANT to have reproductible results during evaluation!
    model.eval()
    model.cuda(args['cuda_num'])


    corpora, class_one_len, class_two_len = read_data(os.path.join(args['text_data_dir'],'%s/'%(dataset_name)))
    all_corpus = corpora[0]+corpora[1]


    print('total number of examples ',len(all_corpus),'\n')
    

    # representations of documnets by averaging all token embeddings
    representations = []

    # representations of documents by using the classification token "<s>" in front of the document 
    representations_cls = []

    with torch.no_grad():
    
        for article in tqdm(all_corpus):
    
            

            tokenized_text = tokenizer.tokenize(article)


            if len(tokenized_text) > 510:
                split_index = len(tokenized_text)//510
#   
                temp_representations = []
                temp_representations_cls = []
                for i in range(split_index+1):
                    

                    temp_tokenized_text = ["<s>"] + tokenized_text[i*510:(i+1)*510] + ["</s>"]

                    indexed_tokens = tokenizer.convert_tokens_to_ids(temp_tokenized_text)
                    tokens_tensor = torch.tensor([indexed_tokens])
                    tokens_tensor = tokens_tensor.cuda(args['cuda_num'])
                    encoded_layers = model(tokens_tensor)[0]
                    output_hidden = encoded_layers.cpu().numpy()


                    temp_representations.append(np.mean(output_hidden[0],axis=0))
                    temp_representations_cls.append(output_hidden[0][0])


                    del tokens_tensor,encoded_layers
                    torch.cuda.empty_cache()
                representations.append(temp_representations)
                representations_cls.append(temp_representations_cls)
            else:
#                
                
                tokenized_text = ["<s>"] + tokenized_text + ["</s>"]

                indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
                tokens_tensor = torch.tensor([indexed_tokens])
                tokens_tensor = tokens_tensor.cuda(args['cuda_num'])
                encoded_layers = model(tokens_tensor)[0]
                output_hidden = encoded_layers.cpu().numpy()

                representations.append(np.mean(output_hidden[0],axis=0))
                representations_cls.append(output_hidden[0][0])

                del tokens_tensor,encoded_layers
                torch.cuda.empty_cache()


    ulti_representations = []
    for representation in representations:
        if type(representation)==list:
            ulti_representations.append(np.mean(representation,axis=0))
        else:
            ulti_representations.append(representation)
            
            
    ulti_representations_cls = []
    for representation in representations_cls:
        if type(representation)==list:
            ulti_representations_cls.append(np.mean(representation,axis=0))
        else:
            ulti_representations_cls.append(representation)

    return ulti_representations, ulti_representations_cls, class_one_len, class_two_len


# for roberta and roberta_hbm, we suggest do not apply any preprocessing
def encode_roberta_hbm(dataset_name):
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # Load pre-trained model (weights)
    model = RobertaModel.from_pretrained('roberta-base')

    # Set the model in evaluation mode to desactivate the DropOut modules
    # This is IMPORTANT to have reproductible results during evaluation!
    model.eval()
    model.cuda(args['cuda_num'])


    corpora, class_one_len, class_two_len = read_data(os.path.join(args['text_data_dir'],'%s/'%(dataset_name)))
    all_corpus = corpora[0]+corpora[1]

    print('size of the dataset:', len(all_corpus))


    segmented_documents = sentences_segmentation(all_corpus,tokenizer)


    length = []
    for i in segmented_documents:
        length.append(len(i))
        
    print('max document length: ', np.max(length))
    print('mean document length: ', np.mean(length))
    print('standard deviation: ', np.std(length))



    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize=(40,10))


    plt.xticks(np.arange(0, max(length)+1, 250.0))
    n, bins,patchs = plt.hist(length, 200, facecolor='g', alpha=0.75)

    plt.show()
 

    ## doc_sen_embeddings: sentence repsentations of all documents shape [num of docs, [num of sentences in a docs, [num of tokens in a sent, 768]]]
    doc_sen_embeddings = []

  
    for doc in tqdm(segmented_documents):



        ## doc_sen_embedding: sentence represetations of a document shape [num of sentences in a doc, 768]
        doc_sen_embedding = []
        for sen in tqdm(doc):
            input_ids = tokenizer(sen)['input_ids']

            # if number of tokens in a sentence is large than 510
            if len(input_ids)>512:
                input_ids = input_ids[:512]

            tokens_tensor = torch.tensor([input_ids]).cuda(args['cuda_num'])
            encoded_layers = model(tokens_tensor)

            embeddings_array = encoded_layers[0][0].cpu().detach().numpy()

            del encoded_layers
            del tokens_tensor

            doc_sen_embedding.append(embeddings_array)
                
            
        doc_sen_embeddings.append(doc_sen_embedding)


  

    ## doc_sen_avg_embeddings: final sentence representations of all documents [num of documents, [num of sentences in a doc, 768]
    doc_sen_avg_embeddings=[]
    for doc in doc_sen_embeddings:

        ## temp_doc shape [num of sentences in a doc, 768]
        temp_doc = []
        for sen in doc:
            avg_sen = np.mean(sen,axis=0)
            temp_doc.append(avg_sen)
        doc_sen_avg_embeddings.append(np.array(temp_doc))





    doc_sen_avg_embeddings = np.array(doc_sen_avg_embeddings)




    doc_dict_neg={}
    for i in range(len(doc_sen_avg_embeddings[:class_one_len])):
        doc_dict_neg[i]=doc_sen_avg_embeddings[i]
        
    doc_dict_pos={}
    for i in range(len(doc_sen_avg_embeddings[class_one_len:])):
        doc_dict_pos[i]=doc_sen_avg_embeddings[class_one_len+i]


   
    return doc_dict_neg, doc_dict_pos


    

if __name__ == "__main__":

    parser = OptionParser(usage='usage: -t encoding_methods (hbm, roberta, fasttext) -d dataset_name')

    
    parser.add_option("-d","--dataset_name", action="store", type="string", dest="dataset_name", help="directory of raw text data", default = 'longer_moviereview')

    parser.add_option('-t', '--encoding_method', type='string', action='callback',dest='encoding_method',callback=list_callback, default=['roberta'])

    (options, _) = parser.parse_args()

        
   

 

    dataset_name = options.dataset_name
    encoding_methods = options.encoding_method


    for value in encoding_methods:
        print(value)
        if value not in ['roberta','hbm','fasttext']:
            parser.error( "Encoding method choice are hbm, fasttext or roberta." )


    for value in encoding_methods:

        if value == 'hbm':
            print('starting coding hbm')
            hbm_roberta_neg, hbm_roberta_pos = encode_roberta_hbm(dataset_name)

            with open(os.path.join(args['data_dir'],'roberta-base_data/%s_neg.p'%(dataset_name)), 'wb') as fp:
                pickle.dump(hbm_roberta_neg, fp)

            with open(os.path.join(args['data_dir'],'roberta-base_data/%s_pos.p'%(dataset_name)), 'wb') as fp:
                pickle.dump(hbm_roberta_pos, fp)

        if value == 'fasttext':

            representations,class_one_len, class_two_len = encode_fasttext(dataset_name)
    
            np.savetxt(os.path.join(args['data_dir'],"fasttext_data/%s_neg.csv"%(dataset_name)), representations[:class_one_len], delimiter=",")
            np.savetxt(os.path.join(args['data_dir'],"fasttext_data/%s_pos.csv"%(dataset_name)), representations[class_one_len:], delimiter=",")
    

        if value == 'roberta':

            representations, representations_cls, class_one_len, class_two_len = encode_roberta(dataset_name)

            np.savetxt(os.path.join(args['data_dir'],"roberta-base_data/%s_neg.csv"%(dataset_name)), representations[:class_one_len], delimiter=",")
            np.savetxt(os.path.join(args['data_dir'],"roberta-base_data/%s_pos.csv"%(dataset_name)), representations[class_one_len:], delimiter=",")

            np.savetxt(os.path.join(args['data_dir'],"roberta-base_data/%s_neg_cls.csv"%(dataset_name)), representations_cls[:class_one_len], delimiter=",")
            np.savetxt(os.path.join(args['data_dir'],"roberta-base_data/%s_pos_cls.csv"%(dataset_name)), representations_cls[class_one_len:], delimiter=",")
