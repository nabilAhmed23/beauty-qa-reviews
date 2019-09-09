import math, heapq
from utilities import utils
import config
import pandas as pd

def get_answers(prod_id, query):
	cleaned_query = utils.process_query(query)
	prod_qa = utils.get_db_data(config.mongo_uri, 27017, config.db_name, config.qa_cleaned_coll, {"asin": prod_id}, projection={'question': 1, 'sentiment_a': 1, 'answer': 1, 'answerType': 1})
	if prod_qa.empty:
		return cleaned_query, 0, None, '?'

	sim_mat = utils.bm25_similarity(cleaned_query, list(prod_qa.loc[:, 'question']))
	index_max = sim_mat.index(max(sim_mat))
	sentiment, answer, answerType = prod_qa.iloc[index_max]['sentiment_a'], prod_qa.iloc[index_max]['answer'], prod_qa.iloc[index_max]['answerType']
	print("RECOMMEND_GET_ANSWERS")
	print("cleaned_query", cleaned_query)
	print("sentiment", sentiment)
	print("answer", answer)
	print("answerType", answerType)
	return cleaned_query, sentiment, answer, answerType


def get_reviews(prod_id, cleaned_query, sentiment, answer_type):
	if answer_type is None or answer_type == '?' or math.isnan(answer_type):
		if sentiment < 0:
			prod_rev = utils.get_db_data(config.mongo_uri, 27017, config.db_name, config.reviews_cleaned_coll, {"asin": prod_id, "overall":{"$gt":3}}, projection={'reviewText': 1, 'og_reviewText': 1, 'overall': 1, '_id': 0, 'helpful': 1})
		elif sentiment > 0:
			prod_rev = utils.get_db_data(config.mongo_uri, 27017, config.db_name, config.reviews_cleaned_coll, {"asin": prod_id, "overall":{"$lt":3}}, projection={'reviewText': 1, 'og_reviewText': 1, 'overall': 1, '_id': 0, 'helpful': 1})
		else:
			prod_rev = utils.get_db_data(config.mongo_uri, 27017, config.db_name, config.reviews_cleaned_coll, {"asin": prod_id}, projection={'reviewText': 1, 'og_reviewText': 1, 'overall': 1, '_id': 0, 'helpful': 1})
	elif answer_type == 'Y':
		prod_rev = utils.get_db_data(config.mongo_uri, 27017, config.db_name, config.reviews_cleaned_coll, {"asin": prod_id, "overall":{"$lt":3}}, projection={'reviewText': 1, 'og_reviewText': 1, 'overall': 1, '_id': 0, 'helpful': 1})
	else:
		prod_rev = utils.get_db_data(config.mongo_uri, 27017, config.db_name, config.reviews_cleaned_coll, {"asin": prod_id, "overall":{"$gt":3}}, projection={'reviewText': 1, 'og_reviewText': 1, 'overall': 1, '_id': 0, 'helpful': 1})

	if not prod_rev.empty:
		review_df = utils.bm25_similarity(cleaned_query, list(prod_rev.loc[:, 'reviewText']))
		index_max_list = heapq.nlargest(3, range(len(review_df)), review_df.take)
		rev = prod_rev.join(pd.DataFrame(review_df, columns=['similarity']))
		print('now',rev)
		# rev = rev[:]['reviewText', 'similarity', 'overall']
		rev = rev.loc[index_max_list]
		rev = rev.to_dict('records')
		print("third", rev)
	else:
		rev = None

	return rev
