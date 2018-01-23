# -*- coding: utf-8 -*-
__author__ = 'Min'

"""

global objects which requires throughout entire program

Flask application for routing and doc2vec model

"""

from flask import Flask
from doc2vec import load_doc2vec_model

flask_application = Flask(__name__)  # flask application
print('loading doc2vec model')
doc2vec_model = load_doc2vec_model('models/doc2vec.model')  # doc2vec model
print('loaded doc2vec model')
