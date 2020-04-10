# pip install google-api-python-client
from googleapiclient.discovery import build

# api keys to perform google search
my_api_key = 'AIzaSyCVUUBPcGR0x2iNRkXJY9NjQIk9zNGVBrA'
my_cse_id = "005209789693038255651:hj63kbdvnmw"


# google search function
def google_search(search_term, api_key, cse_id, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
    return res


# read training titles
training_titles = list()
training_labels = list()
with open('training_task1_b.tsv') as reader:
    for line in reader:
        label, titles_exists, title_body = line.split('\t')
        training_titles.append(title_body.strip())
        training_labels.append(int(label.strip()))

# read test data
test_titles = list()
test_labels = list()
with open('training_task1_b.tsv') as reader:
    for line in reader:
        label, titles_exists, title_body = line.split('\t')
        training_titles.append(title_body.strip())
        training_labels.append(int(label.strip()))


# search training titles in google search to perform similarity on the search returned titles
training_returned_search = list()
for title in training_titles:
    titles_per_search = list()
    results = google_search(title, my_api_key, my_cse_id)
    for result in results['items']:
        if result['title'].endswith('...'):
            titles_per_search.append(result['title'].replace('...', '').strip())
        else:
            titles_per_search.append(result['title'].strip())

    # take the 1st three results
    training_returned_search.append(titles_per_search[:5])


test_returned_search = list()
for title in test_titles:
    titles_per_search = list()
    results = google_search(title, my_api_key, my_cse_id)
    for result in results['items']:
        if result['title'].endswith('...'):
            titles_per_search.append(result['title'].replace('...', '').strip())
        else:
            titles_per_search.append(result['title'].strip())

    # take the 1st three results
    test_returned_search.append(titles_per_search[:3])


# code to perform similarity between title from dataset and titles from search results
