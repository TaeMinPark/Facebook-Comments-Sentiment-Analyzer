# -*- coding: utf-8 -*-
__author__ = 'Min'

"""

train and save Doc2Vec model

"""

from collections import namedtuple
from gensim.models import doc2vec
from konlpy.tag import Twitter
import multiprocessing

twitter = Twitter()
TaggedDocument = namedtuple('TaggedDocument', 'words tags')  # taggeddocument to train doc2vec. (document, (pos or neg))


def read_data(file_name):
    """
    read data from file
    :param file_name: name of file
    :return: array of each line. each lines are splitted by tab.
    """
    with open(file_name, 'r') as file_obj:
        data = [line.split('\t') for line in file_obj.read().splitlines()]
    return data


def tokenize(doc):
    """
    tokenize words

    :param doc: document to tokenize
    :return: tokenized array
    """
    return ['/'.join(twit) for twit in twitter.pos(doc, norm=True, stem=True)]


def get_tagged_documents(data):
    """
    tokenize and return data in format of tagged document which need to train doc2vec.
    :param data: data to train.
    :return: toakenized and taggeddocument data
    """
    return [TaggedDocument(tokenize(row[1]), [row[2]]) for row in data[1:]]

cores = multiprocessing.cpu_count()

vector_size = 300
window_size = 8
train_epoch = 100
min_count = 5
is_dm = 1
seed_num = 1234
iteration_count = 10
workers_count = cores  # workers are count of ores


data = read_data('data/ratings_train.txt')
tagged_train_docs = get_tagged_documents(data)

# doc2vec setup
print('doc2vec setup')
doc_vectorizer = doc2vec.Doc2Vec(size=vector_size, alpha=0.025, min_alpha=0.025, window=window_size,
                                 min_count=min_count, dm=is_dm, seed=seed_num, iter=iteration_count,
                                 workers=workers_count, hs=1)
doc_vectorizer.build_vocab(tagged_train_docs)

print('start training')
for epoch in range(iteration_count):
    doc_vectorizer.train(tagged_train_docs, total_examples=doc_vectorizer.corpus_count, epochs=iteration_count)
    doc_vectorizer.alpha -= 0.002
    doc_vectorizer.min_alpha = doc_vectorizer.alpha
    print('epoch' + str(epoch) + " Finished")

doc_vectorizer.save('models/doc2vec_dm{}.model'.format(str(is_dm)))  # save model
print('finished.')