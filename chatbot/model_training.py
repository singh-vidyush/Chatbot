import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import pickle
import json

import numpy as np
import nltk.data
# nltk.download('punkt')

from nltk.stem import WordNetLemmatizer

from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Dropout
from keras._tf_keras.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

intents = json.load(open('intents.json'))

words = []
classes = []
documents = []
ignore_symbols = ['?','!','.',',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_symbols]
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open('model/words.pkl', 'wb'))
pickle.dump(classes, open('model/classes.pkl', 'wb'))

training = []
output_empty = [0]*len(classes)

for document in documents:
    bag = []
    word_pattern = documents[0]
    word_pattern = [lemmatizer.lemmatize(word) for word in words if word not in ignore_symbols]
    for word in words:
        bag.append(1) if word in word_pattern else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])


random.shuffle(training)

train_x = np.array([np.array(item[0]) for item in training])
train_y = np.array([np.array(item[1]) for item in training])

model = Sequential()
model.add(Dense(128, input_shape = (len(train_x[0]),), activation= 'relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation= 'relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation= 'softmax'))

sgd = SGD(learning_rate=0.01, weight_decay= 1e-6, momentum= 0.9, nesterov= True)

model.compile(loss= 'categorical_crossentropy', optimizer= sgd, metrics= ['accuracy'])
model.fit(np.array(train_x), np.array(train_y), epochs= 200, batch_size= 5, verbose= 1)

model.save('model/chatbot_model.keras')