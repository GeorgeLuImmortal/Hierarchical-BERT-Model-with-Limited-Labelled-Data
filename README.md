# A Sentence-level Hierarchical BERT Model for Document Classification with Limited Labelled Data

This repository is temporarily associated with paper [Lu, J., Henchion, M., Bacher, I. and Mac Namee, B., 2021. A Sentence-level Hierarchical BERT Model for Document Classification with Limited Labelled Data. arXiv preprint arXiv:2106.06738.](https://arxiv.org/pdf/2106.06738.pdf)


### Dependencies
Tested Python 3.6, and requiring the following packages, which are available via PIP:

* Required: [numpy >= 1.16.4](http://www.numpy.org/)
* Required: [scikit-learn >= 0.21.1](http://scikit-learn.org/stable/)
* Required: [pandas >= 1.1.5](https://pandas.pydata.org/)
* Required: [gensim >= 3.7.3](https://radimrehurek.com/gensim/)
* Required: [matplotlib >= 3.3.3](https://matplotlib.org/)
* Required: [torch >= 1.3.1](https://pytorch.org/)
* Required: [transformers >= 4.8.2](https://huggingface.co/transformers/)
* Required: [FastText model trained with Wikipedia 300-dimension](https://fasttext.cc/docs/en/pretrained-vectors.html)
* Required: packaging >= 2.0.0


### Basic Usage

To perform active learning for text labellling, the input corpus of documents should consist of plain text files stored in csv format (two files for one corpus, one for documents belong to class A and one for documents for class B), each row corresponding to one document in that corpus, the format can be refered to the csv file in the sample directory "corpus_data/ProtonPumpInhibitors/".

