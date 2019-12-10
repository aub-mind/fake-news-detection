# -*- coding: utf-8 -*-
"""
@author: Fady Baly 
"""

import glob
import os
import re
from collections import Counter

regex_url = r'(http(s)?:\/\/.)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)'
regex_punk = r'([!\"#\$%\'\(\)\*\+,:;<=>?@\[\\\]\^_`{\|}~—٪’،؟`୍“؛”ۚ【»؛«–‘])'
regex_punk_multi = r'([!\"#\$%\'\(\)\*\+,\.:;<=>?@\[\\\]\^_`{\|}~—٪’،؟`୍“؛”ۚ【»؛«–‘])\1+'


def main():
	testing = list()
	topics_test = list()
	tag = 0
	filenames = glob.glob('task1/fake-news-real-news-eval/*.txt')
	for file in filenames:
		topic = re.sub(r'\d+', '', os.path.split(file)[-1].split('.')[0])
		lines_temp = list()
		with open(file, 'r') as reader:
			for line in reader:
				lines_temp.append(line.strip())

		# get article body
		body = ' '.join(lines_temp)
		body = re.sub(regex_url, 'url', body)
		body = re.sub(regex_punk, r' \g<0> ', body)
		body = re.sub(r'\d+', 'number ', body)
		body = re.sub(regex_punk_multi, r' \1 ', body)
		body = re.sub(r'\s+', ' ', body)

		title_body = 'None'
		# get title, replace with first 10 words from article if title not found
		if len(lines_temp) > 1:
			title = 1
			if len(lines_temp[0].split()) == 1:
				title_body = ' '.join([lines_temp[0], lines_temp[2]])

			else:
				while lines_temp[0] is '':
					lines_temp.remove('')
				title_body = lines_temp[0]
		else:
			title = 0
			title_body = ' '.join(body.split()[:10])

		testing.append([tag, body, title, topic, title_body])
		topics_test.append(topic)

	print('testing topics', Counter(topics_test))

	with open('task1/test_task1_b.tsv', 'w') as writer:
		for tag, body, title, topic, title_body in testing:
			writer.write(topic + '\t' + body + '\n')


if __name__ == '__main__':
	main()
