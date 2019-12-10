# -*- coding: utf-8 -*-
"""
@author: Fady Baly 
"""
import os
import gensim
import numpy as np
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path_to_json_config_file"
model = gensim.models.KeyedVectors.load_word2vec_format('eng_fasttext/crawl-300d-2M-subword.vec')


def check_similarity(entity, dist_dict):
	similarities = list()
	keys = list()
	for key in dist_dict.keys():
		try:
			similarities.append(model.similarity(entity, key))
			keys.append(key)
		except:
			continue
	similar_entity = keys[similarities.index(max(similarities))]
	return dist_dict[similar_entity]


# Instantiates a client
client = language.LanguageServiceClient()

# load data
training_data_path = 'topic/training.tsv'
test_data_path = 'topic/eval_task1_b.tsv'

x_train = list()
y_train = list()
with open(training_data_path, 'r') as reader:
	for line in reader:
		x_train.append(line.split('\t')[-1].strip())
		y_train.append(line.split('\t')[0].strip())

x_test = list()
y_test = list()
with open(test_data_path, 'r') as reader:
	for line in reader:
		x_test.append(line.split('\t')[-1].strip())
		y_test.append(line.split('\t')[0].strip())

# get entities from the google api
entities_train = list()
for index, line in enumerate(x_train):
	print('train', index, line)
	entities_temp = list()
	document = types.Document(
		content=line,
		type=enums.Document.Type.PLAIN_TEXT)
	# Detects the sentiment of the text
	entities = client.analyze_entities(document=document).entities
	for entity in entities:
		entities_temp.append([entity.name, entity.type])
	entities_train.append(entities_temp)

# get entities for test set
entities_test = list()
for index, line in enumerate(x_test):
	print('test', index, line)
	entities_temp = list()
	document = types.Document(
		content=line,
		type=enums.Document.Type.PLAIN_TEXT)

	# Detects the sentiment of the text
	entities = client.analyze_entities(document=document).entities
	for entity in entities:
		entities_temp.append([entity.name, entity.type])
	entities_test.append(entities_temp)


# save extracted entities
np.save('train_entities.npy', np.asarray(entities_train))
np.save('topic/eval_entities.npy', np.asarray(entities_test))

# load train entities
entities_train = np.load('topic/train_entities.npy', allow_pickle=True)
entities_train = np.ndarray.tolist(entities_train)
training_entities_by_topics = dict()
for article_entities, topic in zip(entities_train, y_train):
	if topic not in training_entities_by_topics:
		training_entities_by_topics[topic] = list()
	training_entities_by_topics[topic].append(list(set([entity[0] for entity in article_entities if entity[-1] == 3 or entity[-1] == 6])))

for topic in training_entities_by_topics.keys():
	temp = list()
	for item in training_entities_by_topics[topic]:
		temp += item
	training_entities_by_topics[topic] = list(set(temp))


# load test entities
entities_test = np.load('topic/eval_entities.npy', allow_pickle=True)
entities_test = np.ndarray.tolist(entities_test)
test_entities_by_topics = dict()
for article_entities, topic in zip(entities_test, y_test):
	if topic not in test_entities_by_topics:
		test_entities_by_topics[topic] = list()
	test_entities_by_topics[topic].append(list(set([entity[0] for entity in article_entities if entity[-1] == 3 or entity[-1] == 6])))  # ###########

for topic in test_entities_by_topics.keys():
	temp = list()
	for item in test_entities_by_topics[topic]:
		temp += item
	test_entities_by_topics[topic] = list(set(temp))


# # create distribution vectors for each entity
# get unique entities from the training data
training_set_entities = list()
for article_entities in entities_train:
	training_set_entities += [entity[0] for entity in article_entities if entity[-1] == 3 or entity[-1] == 6]  # ####################################
training_set_entities = list(set(training_set_entities))

# create distribution for each entity
# topics dict order ['sports', 'polit', 'entmt', 'edu', 'tech', 'biz']
dist_dict = dict()
for entity in training_set_entities:
	for topic in training_entities_by_topics.keys():
		if entity not in dist_dict:
			dist_dict[entity] = list()
		dist_dict[entity].append(sum([1 for item in training_entities_by_topics[topic] if entity in item or item in entity]))
		# if topic == 'biz':
		# 	dist_dict[entity] = np.array([probability/max(dist_dict[entity]) for probability in dist_dict[entity]])
	print(entity, dist_dict[entity])


x_train_dist = list()
for index, (article_entities, topic) in enumerate(zip(entities_train, y_train)):
	dist = np.zeros((6,))
	entities = list(set([entity[0] for entity in article_entities if entity[-1] == 3 or entity[-1] == 6]))  # #######################################
	for entity in entities:
		dist += dist_dict[entity]
	if np.max(dist) == 0:
		x_train_dist.append(dist)
		continue
	x_train_dist.append(dist/np.max(dist))


x_test_dist = list()
for index, (article_entities, topic) in enumerate(zip(entities_test, y_test)):
	dist = np.zeros((6,))
	entities = list(set([entity[0] for entity in article_entities if entity[-1] == 3 or entity[-1] == 6]))  # #######################################
	for entity in entities:
		if entity in dist_dict:
			dist += dist_dict[entity]
		else:
			dist = check_similarity(entity, dist_dict)
	if np.max(dist) == 0:
		x_test_dist.append(dist)
		continue
	x_test_dist.append(dist/np.max(dist))

# np.save('train_entities_distribution.npy', np.asarray(x_train_dist))
np.save('topic/eval_entities_distribution.npy', np.asarray(x_test_dist))

# x_train_dist = np.load('topic/train_entities_distribution.npy', allow_pickle=True)
x_test_dis = np.load('topic/eval_entities_distribution.npy', allow_pickle=True)
