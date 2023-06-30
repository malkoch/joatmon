import json
import pickle
import random

import nltk
import numpy as np
import tensorflow as tf
from nltk.stem import WordNetLemmatizer

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)


class GenericAssistant:
    def __init__(self, intents, model_name="assistant_model"):
        self.hist = None
        self.model = None
        self.classes = None
        self.words = None

        self.intents = intents
        self.model_name = model_name

        if intents.endswith(".json"):
            self.load_json_intents(intents)

        self.lemmatizer = WordNetLemmatizer()

    def load_json_intents(self, intents):
        self.intents = json.loads(open(intents).read())

    def train_model(self):
        self.words = []
        self.classes = []
        documents = []
        ignore_letters = ['!', '?', ',', '.']

        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                word = nltk.word_tokenize(pattern)
                self.words.extend(word)
                documents.append((word, intent['name']))
                if intent['name'] not in self.classes:
                    self.classes.append(intent['name'])

        self.words = [self.lemmatizer.lemmatize(w.lower()) for w in self.words if w not in ignore_letters]
        self.words = sorted(list(set(self.words)))

        self.classes = sorted(list(set(self.classes)))

        training = []
        output_empty = [0] * len(self.classes)

        for doc in documents:
            bag = []
            word_patterns = doc[0]
            word_patterns = [self.lemmatizer.lemmatize(word.lower()) for word in word_patterns]
            for word in self.words:
                bag.append(1) if word in word_patterns else bag.append(0)

            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1
            training.append([bag, output_row])

        random.shuffle(training)
        training = np.array(training)

        train_x = list(training[:, 0])
        train_y = list(training[:, 1])

        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.5))
        self.model.add(tf.keras.layers.Dense(64, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.5))
        self.model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))

        sgd = tf.keras.optimizers.legacy.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        self.hist = self.model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=0)

    def save_model(self, model_name=None):
        if model_name is None:
            self.model.save(f"weights/{self.model_name}/weights.h5", self.hist)
            pickle.dump(self.words, open(f'weights/{self.model_name}/words.pkl', 'wb'))
            pickle.dump(self.classes, open(f'weights/{self.model_name}/classes.pkl', 'wb'))
        else:
            self.model.save(f"weights/{model_name}/weights.h5", self.hist)
            pickle.dump(self.words, open(f'weights/{model_name}/words.pkl', 'wb'))
            pickle.dump(self.classes, open(f'weights/{model_name}/classes.pkl', 'wb'))

    def load_model(self, model_name=None):
        if model_name is None:
            self.words = pickle.load(open(f'weights/{self.model_name}/words.pkl', 'rb'))
            self.classes = pickle.load(open(f'weights/{self.model_name}/classes.pkl', 'rb'))
            self.model = tf.keras.models.load_model(f'weights/{self.model_name}/weights.h5')
        else:
            self.words = pickle.load(open(f'weights/{model_name}/words.pkl', 'rb'))
            self.classes = pickle.load(open(f'weights/{model_name}/classes.pkl', 'rb'))
            self.model = tf.keras.models.load_model(f'weights/{model_name}.weights.h5')

    def _clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def _bag_of_words(self, sentence, words):
        sentence_words = self._clean_up_sentence(sentence)
        bag = [0] * len(words)
        for s in sentence_words:
            for i, word in enumerate(words):
                if word == s:
                    bag[i] = 1
        return np.array(bag)

    def _predict_class(self, sentence):
        p = self._bag_of_words(sentence, self.words)
        res = self.model.predict(np.array([p]), verbose=0)[0]
        error_threshold = 0.1
        results = [[i, r] for i, r in enumerate(res) if r > error_threshold]

        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({'intent': self.classes[r[0]], 'probability': r[1]})
        return return_list

    def request(self, message):
        ints = self._predict_class(message)

        return ints[0]['intent'], ints[0]['probability']
