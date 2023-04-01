from __future__ import unicode_literals

from datetime import (
    timedelta
)

import requests

API_URL = 'http://en.wikipedia.org/w/api.php'
RATE_LIMIT = True
RATE_LIMIT_MIN_WAIT = None
RATE_LIMIT_LAST_CALL = timedelta(milliseconds=900)
USER_AGENT = 'wikipedia (https://github.com/goldsmith/Wikipedia/)'


def set_lang(prefix):
    global API_URL
    API_URL = 'http://' + prefix.lower() + '.wikipedia.org/w/api.php'


def summary(title, sentences=1):
    query_params = {'prop': 'info|pageprops', 'inprop': 'url', 'ppprop': 'disambiguation', 'redirects': '', 'titles': title}

    request = _wiki_request(query_params)

    query = request['query']
    pageid = list(query['pages'].keys())[0]
    title = query['pages'][pageid]['title']

    query_params = {'prop': 'extracts', 'explaintext': '', 'titles': title, 'exsentences': sentences}

    request = _wiki_request(query_params)
    return request['query']['pages'][pageid]['extract']


def languages():
    response = _wiki_request({'meta': 'siteinfo', 'siprop': 'languages'})
    return {lang['code']: lang['*'] for lang in response['query']['languages']}


def _wiki_request(params):
    global RATE_LIMIT_LAST_CALL
    global USER_AGENT

    params['format'] = 'json'
    if 'action' not in params:
        params['action'] = 'query'

    r = requests.get(API_URL, params=params, headers={'User-Agent': USER_AGENT})

    return r.json()
