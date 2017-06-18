# Text classification experiment using fastText

## Goal

The goal of text classification is to assign documents (such as emails, posts, text messages, etc) to one or multiple categories (review scores, spam vs non-spam, topics, etc). The dominant approach to build such classificers is ML, that is learning classification rules from examples.

## Data

In order to build such classifiers, we need labeled data, wich consist of documents and their corresponding categories. In this example, we build a classifier which automatically classifies stackexchange questions about cooking into one of several possible tags, such as `pot`, `bowl` or `baking`.

## Classification engine

Facebook AI Research (FAIR) lab [open-sourced](https://github.com/facebookresearch/fastText) fastText on [August 2016](https://code.facebook.com/posts/1438652669495149/fair-open-sources-fasttext/), a library designed to help build scalable solutions for text representation and classification. FastText combines some of the most successful concepts introduced by the natural language processing and machine learning communities in the last few decades.

## Setup

To run the experiment, a python environment and fastText are needed:

Install fastText ([detailed instructions here](https://github.com/facebookresearch/fastText/blob/master/README.md#requirements)):

```
$ git clone https://github.com/facebookresearch/fastText.git
$ cd fastText
$ make
```

Clone the project:

```
$ git clone https://github.com/mpuig/textclassification
$ cd textclassification
```

Create a virtual environment, activate and install python packages:

```
$ python3.6 -m venv venv
$ source venv/bin/activate
$ pip install Cython
$ pip install -r requirements.txt
```

Create the output directory for classification models:

```
$ mkdir models
```

## Getting and preparing the data

As mentioned in the introduction, we need labeled data to train our supervised classifier. In this tutorial, we are interested in building a classifier to automatically recognize the topic of a stackexchange question about cooking. Let's download examples of questions from [the cooking section of Stackexchange](http://cooking.stackexchange.com/), and their associated tags:

```
$ mkdir data
$ wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/cooking.stackexchange.tar.gz
$ tar xvzf cooking.stackexchange.tar.gz -C data
$ head data/cooking.stackexchange.txt
```

Each line of the text file contains a list of labels, followed by the corresponding document. All the labels start by the `__label__` prefix, which is how fastText recognize what is a label or what is a word.  The model is then trained to predict the labels given the word in the document.

Before training our first classifier, we need to split the data into train and validation. We will use the validation set to evaluate how good the learned classifier is on new data.

```
$ wc data/cooking.stackexchange.txt
   15404  169582 1401900 data/cooking.stackexchange.txt
```

Our full dataset contains 15404 examples. Let's split it into a training set of 12404 examples and a validation set of 3000 examples:

```
$ head -n 12404 data/cooking.stackexchange.txt > data/cooking.train
$ tail -n 3000 data/cooking.stackexchange.txt > data/cooking.test
```

## Run the notebook:

```
$ jupyter notebook notebook.ipynb
```
[Open your browser](http://localhost:8888/notebooks/notebook.ipynb)

## Improvements to be done:

- Use (Gensim Phrases)[https://radimrehurek.com/gensim/models/phrases.html#module-gensim.models.phrases]
- Use bigrams
- Apply ideas from [this blog](https://blog.lateral.io/2016/09/fasttext-based-hybrid-recommender/)
- Apply ideas from [this blog](https://bbengfort.github.io/tutorials/2016/05/19/text-classification-nltk-sckit-learn.html)

## Thanks

Some ideas behing the read and write functions, using python generators, comes from [Francesco Bruni code](https://github.com/brunifrancesco/nltk_base/blob/master/2nd.ipynb)
