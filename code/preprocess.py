from utilities import utils
import pandas as pd
from copy import deepcopy
import config

def transform_reviews():
	review_beauty = utils.get_db_data(config.mongo_uri, 27017, config.db_name, config.reviews_coll,projection=None)
	review_beauty_cleaned = review_beauty.copy(deep=True)
	review = review_beauty['reviewText'].tolist()
	review_beauty['og_reviewText'] = review
	cleaned_review = utils.process_text(review)
	i = 1
	for _list in cleaned_review:
		start = ((i - 1) * config.batch_size)
		end = start + len(_list['lemma'])
		review_beauty_cleaned.iloc[start: end]['reviewText'] = _list['lemma']
		review_beauty_cleaned.iloc[start: end]['sentiment'] = _list['sentiment']
		print(review_beauty_cleaned.loc[start])
		i += 1
		utils.store_data(config.mongo_uri, 27017, config.db_name, config.reviews_cleaned_coll+'_test', review_beauty_cleaned)

	print('Data Transformation Complete')


def transform_qa():
	qa_beauty = utils.get_db_data(config.mongo_uri, 27017, config.db_name, config.qa_coll,projection=None)
	qa_beauty_cleaned = qa_beauty.copy(deep=True)

	questions = qa_beauty['question'].tolist()
	answers = qa_beauty['answer'].tolist()

	cleaned_questions = utils.process_text(questions)
	i = 1
	for _list in cleaned_questions:
		start = ((i - 1) * config.batch_size)
		end = start + len(_list['lemma'])
		qa_beauty_cleaned.iloc[start: end]['question'] = _list['lemma']
		qa_beauty_cleaned.iloc[start: end]['sentiment_q'] = _list['sentiment']
		print(qa_beauty_cleaned.loc[start])
		i += 1
		# utils.store_data(config.mongo_uri, 27017, config.db_name, config.qa_cleaned_coll, qa_beauty_cleaned)

	cleaned_answers = utils.process_text(answers)
	i = 1
	for a_list, sentiment_list in enumerate(cleaned_answers):
		start = ((i - 1) * config.batch_size)
		end = start + len(a_list)
		qa_beauty_cleaned.iloc[start: end]['answer'] = _list['lemma']
		qa_beauty_cleaned.iloc[start: end]['sentiment_a'] = _list['sentiment']
		print(qa_beauty_cleaned.loc[start])
		i += 1
		utils.store_data(config.mongo_uri, 27017, config.db_name, config.qa_cleaned_coll, qa_beauty_cleaned)

	print('Data Transformation Complete')


transform_reviews()
