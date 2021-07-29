# A Sentence-level Hierarchical BERT Model for Document Classification with Limited Labelled Data

This repository is temporarily associated with paper [Lu, J., Henchion, M., Bacher, I. and Mac Namee, B., 2021. A Sentence-level Hierarchical BERT Model for Document Classification with Limited Labelled Data. arXiv preprint arXiv:2106.06738.](https://arxiv.org/pdf/2106.06738.pdf)


### Dependencies
Tested Python 3.6, and requiring the following packages, which are available via PIP:

* Required: [numpy >= 1.19.5](http://www.numpy.org/)
* Required: [scikit-learn >= 0.21.1](http://scikit-learn.org/stable/)
* Required: [pandas >= 1.1.5](https://pandas.pydata.org/)
* Required: [gensim >= 3.7.3](https://radimrehurek.com/gensim/)
* Required: [matplotlib >= 3.3.3](https://matplotlib.org/)
* Required: [torch >= 1.9.0](https://pytorch.org/)
* Required: [transformers >= 4.8.2](https://huggingface.co/transformers/)
* Required: [Keras >= 2.0.8](https://keras.io/)
* Required: [Tensorflow >= 1.14.0](https://www.tensorflow.org/)
* Required: [FastText model trained with Wikipedia 300-dimension](https://fasttext.cc/docs/en/pretrained-vectors.html)
* Required: packaging >= 20.0


### Data Processing

The first step is encoding raw text data into different high-dimensional vectorised representations. The raw text data should be stored in directory "raw_corpora/", each dataset should have its individual directory, for example, the "longer_moviereview/" directory under folder "raw_corpora/". The input corpus of documents should consist of plain text files stored in csv format (two files for one corpus, one for documents belong to class A and one for documents for class B). It should be noted that the csv file must be named in the format #datasetname_neg_text.csv or #datasetname_pos_text.csv. Each row corresponding to one document in that corpus, the format can be refered to the csv file in the sample directory "raw_corpora/longer_moviereview/". Then we can start converting the text into vectors:

    python encode_text.py -d dataset_name -t encoding_methods
    
The options of -t is ``hbm'', ``roberta-base'' and ``fasttext''.
