# -*- coding: utf-8 -*-
__author__ = 'Min'

"""

Web routing using flask framework

"""

from global_objects import flask_application, doc2vec_model
from flask import render_template, request
from comments import get_comments
from doc2vec import analyze_comments


@flask_application.route('/', methods=["GET"])
def index():
    return render_template('index.html')


@flask_application.route('/analyze', methods=['POST'])
def analyze():
    access_token = request.form['token']
    post_user_id = request.form['post_user_id']
    post_id = request.form['post_id']

    comments = get_comments(access_token, post_user_id, post_id)
    analyze_result = analyze_comments(doc2vec_model, comments)

    return render_template('analyze.html',
                           positive_count = analyze_result['positive']['count'],
                           negative_count = analyze_result['negative']['count'],
                           positive_most_frequent_words = analyze_result['positive']['most_common_words'],
                           negative_most_frequent_words = analyze_result['negative']['most_common_words']
                           )