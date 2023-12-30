import random

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer

from joatmon.language.intent.core import IntentAgent
from joatmon.nn import (
    Module,
    Tensor
)
from joatmon.nn.layer.activation.relu import ReLU
from joatmon.nn.layer.activation.softmax import Softmax
from joatmon.nn.layer.dropout import Dropout
from joatmon.nn.layer.linear import Linear
from joatmon.nn.loss.cce import CCELoss
from joatmon.nn.optimizer.rmsprop import RMSprop

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)


class IntentModel(Module):
    def __init__(self, in_shape, out_shape):
        super(IntentModel, self).__init__()

        self.linear1 = Linear(in_shape, 128)
        self.relu1 = ReLU()
        self.dropout1 = Dropout(.5)
        self.linear2 = Linear(128, 64)
        self.relu2 = ReLU()
        self.dropout2 = Dropout(.5)
        self.linear3 = Linear(64, out_shape)
        self.softmax = Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.linear2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.linear3(x)
        x = self.softmax(x)

        return x


class LocalIntent(IntentAgent):
    """
    Deep Deterministic Policy Gradient

    # Arguments
        actor_model (`keras.nn.Model` instance): See [Model](#) for details.
        critic_model (`keras.nn.Model` instance): See [Model](#) for details.
        optimizer (`keras.optimizers.Optimizer` instance):
        See [Optimizer](#) for details.
        action_inp (`keras.layers.Input` / `keras.layers.InputLayer` instance):
        See [Input](#) for details.
        tau (float): tau.
        gamma (float): gamma.
    """

    def __init__(self, intents):
        self.hist = None
        self.model = None
        self.classes = None
        self.words = None

        self.intents = intents

        self.lemmatizer = WordNetLemmatizer()

        self.words = []
        self.classes = []
        ignore_letters = ['!', '?', ',', '.']

        for intent in self.intents:
            for pattern in intent['patterns']:
                word = nltk.word_tokenize(pattern)
                self.words.extend(word)
                if intent['name'] not in self.classes:
                    self.classes.append(intent['name'])

        self.words = [self.lemmatizer.lemmatize(w.lower()) for w in self.words if w not in ignore_letters]
        self.words = sorted(list(set(self.words)))

        self.classes = sorted(list(set(self.classes)))

        self.model = IntentModel(len(self.words), len(self.classes))

    def train_model(self):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """

        documents = []

        for intent in self.intents:
            for pattern in intent['patterns']:
                word = nltk.word_tokenize(pattern)
                documents.append((word, intent['name']))
                if intent['name'] not in self.classes:
                    self.classes.append(intent['name'])

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

        train_x = [x[0] for x in training]
        train_y = [x[1] for x in training]

        self.model.train()

        # optimizer = Adam(params=list(self.model.parameters()), lr=.0001, weight_decay=1e-6)
        optimizer = RMSprop(params=list(self.model.parameters()), lr=.0005, weight_decay=1e-6)
        loss = CCELoss()

        batch_x = Tensor.from_array(train_x)
        batch_y = Tensor.from_array(train_y)
        for epoch in range(200):
            predict = self.model(batch_x)

            model_loss = loss(predict, batch_y)
            optimizer.zero_grad()
            model_loss.backward()
            optimizer.step()

    def _clean_up_sentence(self, sentence):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def _bag_of_words(self, sentence, words):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        sentence_words = self._clean_up_sentence(sentence)
        bag = [0] * len(words)
        for s in sentence_words:
            for i, word in enumerate(words):
                if word == s:
                    bag[i] = 1
        return np.array(bag)

    def _predict_class(self, sentence):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        self.model.eval()

        p = self._bag_of_words(sentence, self.words)
        res = self.model(Tensor.from_array(np.array([p])))[0]
        error_threshold = 0.001
        results = [[i, r] for i, r in enumerate(res) if r.data > error_threshold]

        results.sort(key=lambda x: x[1].data, reverse=True)
        return_list = []
        for r in results:
            return_list.append({'intent': self.classes[r[0]], 'probability': r[1].data})
        return return_list

    def request(self, message):
        """
        Remember the transaction.

        Accepts a state, action, reward, next_state, terminal transaction.

        # Arguments
            transaction (abstract): state, action, reward, next_state, terminal transaction.
        """
        ints = self._predict_class(message)

        return ints[0]['intent'], ints[0]['probability']
