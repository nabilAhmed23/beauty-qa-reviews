from pymongo import MongoClient
import pandas as pd
import re, os
import nltk
from nltk.stem import WordNetLemmatizer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from rank_bm25 import BM25Okapi
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import config

analyser = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()
spacy_nlp = spacy.load('en_core_web_sm')

def get_db_data(host, port, db_name, collection, query={}, projection={'reviewText': 1, 'question': 1, 'answer': 1}, df=True):
	print('get_db_data')
	client = MongoClient(host=host, port=port)
	db = client[db_name]
	data = list(db[collection].find(query,projection))
	if not df:
		return data
	data_df = pd.DataFrame(data)
	return data_df


def get_db_distinct(host, port, db_name, collection, field, query={}):
	print('get_db_distinct')
	client = MongoClient(host=host, port=port)
	db = client[db_name]
	data = db[collection].distinct(field,query)
	return data


def clean_text(line):
	line = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~]', "", line)
	line = re.sub('\s+', ' ', line)
	line = re.sub("\'", "", line)
	line = re.sub("\"", "", line)
	line = line.lower()
	return line


def sentiment_scores(sentence):
	sentiment = analyser.polarity_scores(sentence)
	# print("{:-<40} {}".format(sentence, str(sentiment)))
	return round(sentiment['compound'], 1)


def lemmatize_text(text):
	spacy_text = spacy_nlp(text)
	token_list = [token.text for token in spacy_text if not token.is_stop]
	token_str = ' '.join(token_list)
	lemma = [lemmatizer.lemmatize(wt) for wt in nltk.word_tokenize(token_str)]
	return lemma


def process_text(text):
	print('process_text')
	lemmatizer = WordNetLemmatizer()
	lemma_list = []
	sentiment_list = []
	txt_len = len(text)

	idx = 1
	for line in text:
		sentiment = sentiment_scores(line)
		sentiment_list.append(sentiment)
		line = clean_text(line)
		lemma = lemmatize_text(line)
		lemma_list.append(' '.join(lemma))

		if idx % config.batch_size == 0 or idx == txt_len:
			yield {'lemma': lemma_list, 'sentiment': sentiment_list}
			lemma_list = []
			sentiment_list = []

		idx += 1


def process_query(query):
	print('process_query')
	spacy_nlp = spacy.load('en_core_web_sm')
	lemma_list = []
	query = clean_text(query)

	spacy_line = spacy_nlp(query)
	token_list = [token.text for token in spacy_line if not token.is_stop]
	token_str = ' '.join(token_list)

	lemma = [lemmatizer.lemmatize(wt) for wt in nltk.word_tokenize(token_str)]
	return (' '.join(lemma))


def store_data(host, port, db_name, collection, data):
	client = MongoClient(host=host, port=port)
	db = client[db_name]
	for idx, item in data.iterrows():
		db[collection].update({"_id": item["_id"]}, {"$set": item.to_dict()}, upsert=True)


def cos_similarity(query, corpus):
	corpus.append(query)
	tfidf = TfidfVectorizer()
	tfidf_vectors = tfidf.fit_transform(corpus)
	sim_matrix = cosine_similarity(tfidf_vectors[-1], tfidf_vectors[:-1])
	return sim_matrix[0]


def bm25_similarity(query, corpus):
	tokenized_corpus = [doc.split(" ") for doc in corpus]
	bm25 = BM25Okapi(tokenized_corpus)
	tokenized_query = query.split(" ")
	scores = bm25.get_scores(tokenized_query)
	return scores


def save_vec(vectorizer, path):
	joblib.dump(vectorizer, path)


def load_vec(path):
	return joblib.load(path)
