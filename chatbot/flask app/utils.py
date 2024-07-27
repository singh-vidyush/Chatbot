import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import pickle
import json

import numpy as np
import nltk.data
# nltk.download('punkt')

from nltk.stem import WordNetLemmatizer

from keras._tf_keras.keras.models import load_model

def clean_up_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    ignore_symbols = ['?','!','.',',']

    sentence_word = nltk.word_tokenize(sentence)
    sentence_word = [lemmatizer.lemmatize(word) for word in sentence_word if word not in ignore_symbols]

    return sentence_word


def bag_of_words(sentence):
    words = pickle.load(open('flask app/model/words.pkl', 'rb'))

    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)

    for w in sentence_words:
        for i, w in enumerate(words):
            if words == w:
                bag[i] = 1

    return np.array(bag)


def prediction_class(sentence):
    classes = pickle.load(open('flask app/model/classes.pkl', 'rb'))
    model = load_model('flask app/model/chatbot_model.keras')

    bow = bag_of_words(sentence)
    bow_array = np.array([bow])

    res = model.predict(bow_array)[0]
    ERROR_THRESHOLD = 0.25

    result = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    result.sort(key= lambda x: x[1], reverse= True)

    result_list = []
    for r in result:
        result_list.append({'intent': classes[r[0]], 'probability': str(r[1])})

    return result_list


def get_response(intent_list):
    intents_json = json.load(open('flask app/model/intents.json'))

    tag = intent_list[0]['intent']
    list_of_intents = intents_json['intents']

    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break

    return result