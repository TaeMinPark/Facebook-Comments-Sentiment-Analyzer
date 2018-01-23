# -*- coding: utf-8 -*-
__author__ = 'Min'

"""

Functions that requires to download comments from facebook graph API.

Facebook access token, post ID, and user ID of the post should be given.

You can get a token from here: https://developers.facebook.com/tools/explorer/

"""

import requests

GRAPH_API_VERSION = 'v2.11'


def get_comments(access_token, post_user_id, post_id):
    """
    get comments from a facebook post until there is no more comments left

    :param access_token: access token to call graph api
    :param post_user_id: facebook user id of the post
    :param post_id: facebook post id
    :return: comments array
    """

    request_url = 'https://graph.facebook.com/{}/{}_{}/comments?access_token={}'\
        .format(GRAPH_API_VERSION, post_user_id, post_id, access_token)
    request_obj = requests.get(request_url)

    comments = []
    while True:
        request_data = request_obj.json()  # json to arrays

        # if error
        if "error" in request_obj:
            raise Exception(request_data['error']['message'])

        for comment in request_data['data']:
            # line break to a single tab
            message = comment['message'].replace('\n', '\t')
            comments.append(message)

        # more comments?
        if 'paging' in request_data and 'next' in request_data['paging']:
            request_obj = requests.get(request_data['paging']['next'])
        else:
            # return comments when there is no more comments left
            return comments
