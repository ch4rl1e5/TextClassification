import csv
import pandas as pd
import spacy

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
from bs4 import BeautifulSoup
from unidecode import unidecode

nlp = spacy.load("pt_core_news_sm")


class Trainer:
    def __init__(self):
        self.train_data = []
        self.test_data = []
        self.labels = []
        self.labeled_data_sets = []
        self.tokenizer = Tokenizer(num_words=20000)
        self.keras = Sequential()

        self.stop_words = spacy.lang.pt.stop_words.STOP_WORDS
        self.stop_words = [self.remove_accented_chars(row) for row in self.stop_words]

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

    # Remove tags HTML
    def strip_html_tags(self, text):
        soup = BeautifulSoup(text, "html.parser")
        stripped_text = soup.get_text(separator=" ")
        return stripped_text

    # Remove acentos do texto, e.g. café
    def remove_accented_chars(self, text):
        text = unidecode(text)
        return text

    def remove_stop_words(self, text):
        print('Removing stop words... \n')

        spacy.lang.pt.stop_words.STOP_WORDS.add('o')
        nlp.vocab['o'].is_stop = True

        spacy.lang.pt.stop_words.STOP_WORDS.add('a')
        nlp.vocab['a'].is_stop = True

        spacy.lang.pt.stop_words.STOP_WORDS.add('e')
        nlp.vocab['e'].is_stop = True

        spacy.lang.pt.stop_words.STOP_WORDS.add('bom')
        nlp.vocab['bom'].is_stop = False

        doc = nlp(text)
        result = [token.text if not token.is_stop else '' for token in doc]
        s = ' '
        s = s.join(result)
        print(s)
        return s

    def lemmatizate(self, text):
        print('Lemmarization... \n')
        doc = nlp(text)
        result = [word.lemma_ if word.lemma_ != '-PRON-' else word.lower_ for word in doc]
        s = ' '
        s = s.join(result)
        print(s)
        return s

    def remove_special_characters(self, text):
        result = str()
        for char in text:
            if char.isalnum() or char == ' ':
                result = result + char
        print(result)
        return result

    def clear_sentence(self, text):
        text = self.strip_html_tags(text)
        text = self.lemmatizate(text)
        text = self.remove_accented_chars(text)
        text = self.remove_stop_words(text)
        text = self.remove_special_characters(text)

        return text.upper()

    def preprocess(self, df):
        if not path.exists("datasets/clean_dataset.csv"):
            sentences = df['sentences'].values
            y = df['labels'].values
            clean_sentences = [self.clear_sentence(row) for row in sentences]
            dataset_df = pd.DataFrame(list(zip(clean_sentences, y)), columns=['sentences', 'labels'])
            dataset_df.to_csv(r'datasets/clean_dataset.csv', index=False, header=True)

        with open('datasets/clean_dataset.csv', encoding='UTF-8', newline='') as csvfile:
            data = list(csv.reader(csvfile, delimiter=','))
            del data[0]
            clean_df = pd.DataFrame(data, columns=['sentences', 'labels'])
            clean_sentences = clean_df['sentences'].values
            y = clean_df['labels'].values

        self.tokenizer.fit_on_texts(clean_sentences)
        x_dataset = self.tokenizer.texts_to_sequences(clean_sentences)

        x_train, x_test, y_train, y_test = train_test_split(x_dataset, y, test_size=0.3,
                                                            random_state=1000)

        x_train = pad_sequences(x_train, padding='post', maxlen=100)
        x_test = pad_sequences(x_test, padding='post', maxlen=100)

        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        return x_train, x_test, y_train, y_test

    def train_keras(self, df):

        x_train, x_test, y_train, y_test = self.preprocess(df)

        print(x_train, x_test, y_train, y_test)

        if not path.exists("sentiment.h5"):
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
        input = pad_sequences(
            self.tokenizer.texts_to_sequences([self.clear_sentence(text)]), padding='post',
            maxlen=100)
        model = load_model('sentiment.h5')
        output = model.predict_classes(input)
        for i in output:
            print(self.labels[i])
