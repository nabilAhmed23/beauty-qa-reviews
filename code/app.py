from flask import request, Flask, jsonify
import main, recommend
from utilities import utils

app = Flask(__name__)

@app.route('/')
def home():
	return jsonify(message="Success")


@app.route('/query', methods=['POST', 'OPTIONS'])
def process_query():
	req_body = request.get_json()
	print(req_body)
	query = req_body['query']
	prod_id = req_body['prod_id']
	prod_url = 'www.amazon.com/dp/' + prod_id
	data = {'product_url':prod_url}
	cleaned_query, sentiment, answer, answerType = recommend.get_answers(prod_id, query)
	if answer is None:
		data['answer']='No relevant answer found'
	else:
		data['answer_sentiment']= sentiment
		data['answer']= answer
	reviews = recommend.get_reviews(prod_id, cleaned_query, sentiment, answerType)
	if reviews is None:
		data['reviews'] = 'No reviews'
	else:
		data['reviews'] = reviews
	return jsonify(data)


if __name__ == '__main__':
	app.run(host="localhost", port=5000, debug=True)
