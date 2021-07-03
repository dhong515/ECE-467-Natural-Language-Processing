# NLP Project 3
# Ravindra Bisram, Danny Hong
# https://towardsdatascience.com/multi-class-text-classification-with-lstm-using-tensorflow-2-0-d88627c10a35
# Multi Class Text Classification with LSTM using TensorFlow 2.0

import os
import csv
# Suppress all that annoying tf logging: https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import sklearn
import nltk

nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))

print(tf.__version__)

vocab_size = 5000
embedding_dim = 64
max_length = 200
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = .8
testing_portion = .5

articles = []
labels = []

# Obtain relevant files from user
default_path = "TC_provided"
train_filename = str(input("Enter training file: ") or default_path + "/corpus1_train.labels")

texts = []
classes = []
filenames = []
with open(train_filename) as fp:
    for line in fp:
        path_name = line.split(" ")[0]
        classification = line.split(" ")[1]
        classification = classification.rstrip("\n")
        full_path = default_path + path_name
        filenames.append(full_path)
        # print(path_name)
        with open(full_path) as fp2:
            # article = fp2.readlines()[0]
            article = fp2.read().replace('\n', '')
            # print(article)
            for word in STOPWORDS:
                token = ' ' + word + ' '
                article = article.replace(token, ' ')
                article = article.replace(' ', ' ')
            texts.append(article)
            # print(len(article))
            # for sentence in fp2:
            #    texts.append(sentence)
        classes.append(classification)

"""
print(len(texts))
print(len(classes))
print(texts[0])
"""
articles_preshuffle = texts
labels_preshuffle = classes

articles, labels = sklearn.utils.shuffle(articles_preshuffle, labels_preshuffle)

train_size = int(len(articles) * training_portion)

train_articles = articles[0: train_size]
train_labels = labels[0: train_size]

validation_articles = articles[train_size:]
validation_labels = labels[train_size:]

"""
print(train_size)
print(len(train_articles))
print(len(train_labels))
print(len(validation_articles))
print(len(validation_labels))
"""

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_articles)
word_index = tokenizer.word_index
# print(dict(list(word_index.items())[0:10]))

train_sequences = tokenizer.texts_to_sequences(train_articles)
# print(train_sequences[10])

train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
"""
print(len(train_sequences[0]))
print(len(train_padded[0]))

print(len(train_sequences[1]))
print(len(train_padded[1]))

print(len(train_sequences[10]))
print(len(train_padded[10]))

print(train_padded[10])
"""

validation_sequences = tokenizer.texts_to_sequences(validation_articles)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# print(len(validation_sequences))
# print(validation_padded.shape)

label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)

training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))
"""
print(training_label_seq[0])
print(training_label_seq[1])
print(training_label_seq[2])
print(training_label_seq.shape)

print(validation_label_seq[0])
print(validation_label_seq[1])
print(validation_label_seq[2])
print(validation_label_seq.shape)
"""

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_article(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


# print(decode_article(train_padded[10]))
# print('---')
# print(train_articles[10])

model = tf.keras.Sequential([
    # Add an Embedding layer expecting input vocab of size 5000, and output embedding dimension of size 64 we set at the top
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
    # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    # use ReLU in place of tanh function since they are very good alternatives of each other.
    tf.keras.layers.Dense(embedding_dim, activation='relu'),
    # Add a Dense layer with 6 units and softmax activation.
    # When we have multiple outputs, softmax convert outputs layers into a probability distribution.
    tf.keras.layers.Dense(6, activation='softmax')
])
model.summary()

print(set(labels))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
num_epochs = 10
history = model.fit(train_padded, training_label_seq, epochs=num_epochs, validation_data=(validation_padded, validation_label_seq), verbose=2)


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


# plot_graphs(history, "accuracy")
# plot_graphs(history, "loss")


# NOW WE ARE GOING TO ADD THE TEST SET
test_filename = str(input("\nEnter testing file: ") or default_path + "/corpus1_test.labels")
test_texts = []
test_classes = []
with open(train_filename) as fp:
    for line in fp:
        path_name = line.split(" ")[0]
        classification = line.split(" ")[1]
        classification = classification.rstrip("\n")
        full_path = default_path + path_name
        # print(path_name)
        with open(full_path) as fp2:
            # article = fp2.readlines()[0]
            article = fp2.read().replace('\n', '')
            # print(article)
            for word in STOPWORDS:
                token = ' ' + word + ' '
                article = article.replace(token, ' ')
                article = article.replace(' ', ' ')
            test_texts.append(article)
            # print(len(article))
            # for sentence in fp2:
            #    texts.append(sentence)
        test_classes.append(classification)

test_sequences = tokenizer.texts_to_sequences(test_texts)
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

test_label_seq = np.array(label_tokenizer.texts_to_sequences(test_classes))

# test_results = model.evaluate(test_padded, test_label_seq, verbose=1)
test_results = model.predict(test_padded)
# print(test_results[:5])
test_results = np.argmax(test_results, axis = 1)
# label = np.argmax(test_label_seq,axis = 1)[:5]
flat_list = np.array([item for sublist in test_label_seq for item in sublist])

# print(test_results)
# print(label)
# print(flat_list)

accuracy = test_results == flat_list

# print(accuracy)
# accuracy = int(accuracy == 'true')
final_acc = sum(accuracy)/len(accuracy)
print("Accuracy Rate: ", final_acc)
