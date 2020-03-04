import csv
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras import layers
from keras.models import load_model
from os import path
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


class Trainer:
    def __init__(self):
        self.train_data = []
        self.test_data = []
        self.labels = []
        self.labeled_data_sets = []
        self.tokenizer = Tokenizer(num_words=20000)

    def get_data(self):

        with open('datasets/atendimentos_sentimentos.csv', encoding='UTF-8', newline='') as csvfile:
            data = list(csv.reader(csvfile, delimiter=','))
            self.get_labels(data, 1, True)
            text_rows = self.clear_columns(data, True)
            df = pd.DataFrame(text_rows, columns=['labels', 'sentences'])
        return df

    def clear_columns(self, data, has_title=False):
        if has_title:
            del data[0]

        return [[self.labels.index(row[1]), row[0]] for row in data]

    def get_labels(self, data, col, has_title=False):
        if has_title:
            del data[0]

        labels = []
        for row in data:
            labels.append(row[col])

        self.labels = list(dict.fromkeys(labels))

    def train_lr(self, df):
        sentences = df['sentences'].values
        y = df['labels'].values
        sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.3,
                                                                            random_state=1000)

        self.vectorizer = CountVectorizer()
        self.vectorizer.fit(sentences_train)

        x_train = self.vectorizer.transform(sentences_train)
        x_test = self.vectorizer.transform(sentences_test)

        self.classifier = LogisticRegression()
        self.classifier.fit(x_train, y_train)
        score = self.classifier.score(x_test, y_test)

        print("Accuracy:", score)

    def train_keras(self, df):
        sentences = df['sentences'].values
        y = df['labels'].values
        sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.3,
                                                                            random_state=1000)

        self.tokenizer.fit_on_texts(sentences_train)

        x_train = self.tokenizer.texts_to_sequences(sentences_train)
        x_test = self.tokenizer.texts_to_sequences(sentences_test)

        x_train = pad_sequences(x_train, padding='post', maxlen=100)
        x_test = pad_sequences(x_test, padding='post', maxlen=100)

        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        if not path.exists("sentiment.h5"):
            self.keras = Sequential()
            self.keras.add(layers.Embedding(input_dim=len(self.tokenizer.word_index) + 1,
                                            output_dim=50,
                                            input_length=100))
            self.keras.add(layers.Flatten())
            self.keras.add(layers.Dense(64, activation='relu'))
            self.keras.add(layers.Dense(len(self.labels), activation='softmax'))

            self.keras.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            self.keras.summary()

            self.keras.fit(x_train, y_train, epochs=40, verbose=True, validation_data=(x_test, y_test),
                           batch_size=2000, shuffle=True)

            loss, accuracy = self.keras.evaluate(x_train, y_train, verbose=False)
            print("Training Accuracy: {:.4f}".format(accuracy))
            loss, accuracy = self.keras.evaluate(x_test, y_test, verbose=False)
            print("Test Accuracy: {:.4f}".format(accuracy))

            self.keras.save('sentiment.h5')

    def bow(self, text):
        input = pad_sequences(self.tokenizer.texts_to_sequences([text]), padding='post', maxlen=100)
        model = load_model('sentiment.h5')
        output = model.predict_classes(input)
        for i in output:
            print(self.labels[i])
