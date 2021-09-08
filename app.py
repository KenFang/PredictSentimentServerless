import ujson
import pickle
import re
from string import punctuation
from nltk.stem import WordNetLemmatizer
from nltk.util import everygrams

import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def extract_feature(document):
    nltk.data.path = ['./nltk_data/']

    stopwords_eng = stopwords.words('english')

    lemmatizer = WordNetLemmatizer()

    words = word_tokenize(document)
    lemmas = [str(lemmatizer.lemmatize(w)) for w in words if w not in stopwords_eng and w not in punctuation]
    document = " ".join(lemmas)
    document = document.lower()
    document = re.sub(r'[^a-zA-Z0-9\s]', ' ', document)
    words = [w for w in document.split(" ") if w != "" and w not in stopwords_eng and w not in punctuation]
    return [str(" ".join(ngram)) for ngram in list(everygrams(words, max_len=3))]


def bag_of_words(document):
    bag = {}
    for w in document:
        bag[w] = bag.get(w, 0) + 1
    return bag


def get_predicate_model():
    model_file = open('./sa_classifier.pickle', 'rb')
    model = pickle.load(model_file)
    model_file.close()
    return model


def get_sentiment(review):
    words = extract_feature(review)
    words = bag_of_words(words)
    model = get_predicate_model()
    return model.classify(words)


def lambda_handler(event, context):
    body = ujson.loads(event["body"])

    try:
        return {
            "statusCode": 200,
            "body": ujson.dumps({
                "message": "The Movie Review is.....",
                "positive/ negative ?": get_sentiment(body["reviews"])
            })
        }
    except Exception as err:
        return {
            "statusCode": 400,
            "body": ujson.dumps({
                "message": "Something went wrong. Unable to parse data !",
                "error": str(err)
            })
        }

