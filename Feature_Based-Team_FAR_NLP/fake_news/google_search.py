# -*- coding: utf-8 -*-
"""
@author: Fady Baly 
"""

# pip install google-api-python-client
from googleapiclient.discovery import build
import numpy as np
import pandas as pd
# api keys to perform google search
my_api_key = 'api_key'
my_cse_id = "cse_id"


# google search function
def google_search(search_term, api_key, cse_id, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
    return res


# read eval data
eval_titles = list()
eval = pd.read_csv("task1/eval_task1_b.tsv", sep="\t", quoting=3, names=['labels', 'titles', ''], header=None)
for titles, labels in zip(eval['titles'], eval['labels']):
    eval_titles.append(titles.strip())


eval_returned_search = list()
for index, title in enumerate(eval_titles):
    titles_per_search = list()
    results = google_search(title, my_api_key, my_cse_id)
    print(index, results.keys(), title)
    for result in results['items']:
        if result['title'].endswith('...'):
            titles_per_search.append(result['title'].replace('...', '').strip())
        else:
            titles_per_search.append(result['title'].strip())

    # take the 1st three results
    eval_returned_search.append(titles_per_search[:3])

np.save('task1/eval_search_results', np.asarray(eval_returned_search))
x = np.ndarray.tolist(np.load('task1/eval_search_results.npy'))
