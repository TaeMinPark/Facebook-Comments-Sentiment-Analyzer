# -*- coding: utf-8 -*-
__author__ = 'Min'

"""

Doc2Vec utilities.

Requires pre-trained model

"""

from konlpy.tag import Twitter  # To morphologic analyze
from gensim.models import Doc2Vec
import nltk


twitter = Twitter()


def tokenize(doc):
    """
    Tokenize document

    ex) 가다/동사
    :param doc: document to tokenize
    :return: tokenized document array
    """
    return ['/'.join(twit) for twit in twitter.pos(doc, norm=True, stem=True)]


def load_doc2vec_model(path):
    """
    load pre-trained doc2vec model
    :param path: path of pre-trained doc2vec model
    :return: doc2vec object
    """
    # load train data
    return Doc2Vec.load(path)


def analyze_single_comment(model, tokenized_comment):
    """
    analyze single comment if this is positive or negative.

    :param model: doc2vec model
    :param tokenized_comment: comment to analyze
    :return: result. '0': Negative, '1': Positive
    """
    new_vector = model.infer_vector(tokenized_comment)
    sims = model.docvecs.most_similar([new_vector])
    return sims[0][0]


def analyze_comments(model, comments):
    """
    analyze if this comments are positive or negative. And return results.

    :param model: doc2vec model
    :param comments: comments to analyze
    :return: result dictionary. ex)

    {
        'positive': {'count': 3, "most_common_words": [('./Punctuation', 68630), ('영화/Noun', 51365), ('하다/Verb', 50281)]},
        'negative': {'count': 2, "most_common_words": [('./Punctuation', 68630), ('영화/Noun', 51365), ('하다/Verb', 50281)]}
    }
    """
    negative_words_tokens = []
    positive_words_tokens = []
    positive_count = 0
    negative_count = 0

    for comment in comments:
        tokenized_comment = tokenize(comment)
        result = analyze_single_comment(model, tokenized_comment)
        if result == '0':
            # negative
            negative_count += 1
            for token in tokenized_comment:
                negative_words_tokens.append(token)

        else:
            # positive
            positive_count += 1
            for token in tokenized_comment:
                positive_words_tokens.append(token)

    return {
        'positive':
            {
                'count': positive_count,
                'most_common_words': nltk.Text(positive_words_tokens, name='NMSC').vocab().most_common(10)
            },
        'negative':
            {
                'count': negative_count,
                'most_common_words': nltk.Text(negative_words_tokens, name='NMSC').vocab().most_common(10)
            }
    }
