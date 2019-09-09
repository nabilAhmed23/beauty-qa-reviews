import os
os.system('pip install -r requirements.txt')
os.system('python -m spacy download en_core_web_sm')
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('vader_lexicon')
# nltk.download()
