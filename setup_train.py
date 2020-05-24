# import modules needeed
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import asarray
from numpy import zeros
import nltk
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Embedding, LSTM
from keras.layers.core import Dense, Dropout
from keras.callbacks import ReduceLROnPlateau

# download stopwords (we're gonna need it later)
nltk.download('stopwords')
from nltk.corpus import stopwords

# load data
print("Loading data...")
df = pd.read_csv("./data/training.1600000.processed.noemoticon.csv", encoding = 'latin', header = None)

# change columns names for reference
df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']
# drop useless columns
df = df.drop(['id', 'date', 'flag', 'user'], axis = 1)

# show the first five rows of data (to check if everything's fine)
print(df.head())

# count dataset samples to know how many of each class we have
print("Data distribution: ")
print(df.target.value_counts())

# clean text to remove users, links and stopwords and then split it in tokens
def clean_text(text):
    text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
    text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
    
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    
    return " ".join(tokens)

# apply clean_text function in our data
print("Cleaning data...")
df.text = df.text.apply(lambda x: clean_text(x))

# show the first five rows of data (to verify again)
print("Cleaned data: ")
print(df.head())

# set 80% for our data to train
train_size = 0.8

# split our data into train set (80%) and test set (20%)
train_data, test_data = train_test_split(df, test_size = 1 - train_size, random_state = 0, stratify = df.target)

# length of each set
print("Train data size: ", len(train_data))
print("Test data size: ", len(test_data))

# how many examples of each class there is in each set
print("Train data distr: ")
print(train_data.target.value_counts())
print("Test data distr: ")
print(test_data.target.value_counts())

# get max length of the train data
max_length = max([len(s.split()) for s in train_data.text])

# create a label encoder
encoder = LabelEncoder()
# enconde labels (0 or 1) in train data
encoder.fit(train_data.target.to_list())

# transform labels in y_train and y_test data to the encoded ones
y_train = encoder.transform(train_data.target.to_list())
y_test = encoder.transform(test_data.target.to_list())

# reshape y_train and y_test data
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# create a tokenizer
tokenizer = Tokenizer()
# fit the tokenizer in the train text
tokenizer.fit_on_texts(train_data.text)

# saving tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol = pickle.HIGHEST_PROTOCOL)

# pad sequences in x_train data set to the max length
x_train = pad_sequences(tokenizer.texts_to_sequences(train_data.text),
                        maxlen = max_length)
# pad sequences in x_test data set to the max length
x_test = pad_sequences(tokenizer.texts_to_sequences(test_data.text),
                       maxlen = max_length)

print("x_train shape: ", x_train.shape)
print("x_test shape: ", x_test.shape)

print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# load embedding as a dict
def load_embedding(filename):
    # load embedding into memory, skip first line
    file = open(filename,'r',encoding="utf-8")
    lines = file.readlines()
    file.close()
    # create a map of words to vectors
    embedding = dict()
    for line in lines:
        parts = line.split()
        # key is string word, value is numpy array for vector
        embedding[parts[0]] = asarray(parts[1:], dtype='float32')
    return embedding

# create a weight matrix for the Embedding layer from a loaded embedding
def get_weight_matrix(embedding, vocab):
    # total vocabulary size plus 0 for unknown words
    vocab_size = len(vocab) + 1
    # define weight matrix dimensions with all 0
    weight_matrix = zeros((vocab_size, embedding_dim))
    # step vocab, store vectors using the Tokenizer's integer mapping
    for word, i in vocab.items():
        vector = embedding.get(word)
        if vector is not None:
            weight_matrix[i] = vector
    return weight_matrix

# contains the index for each word
vocab = tokenizer.word_index
# total number of words in our vocabulary, plus one for unknown words
vocab_size = len(tokenizer.word_index) + 1
# embedding dimensions
embedding_dim = 200

# load embedding from file
raw_embedding = load_embedding('./data/glove/glove.twitter.27B.200d.txt')
# get vectors in the right order
embedding_matrix = get_weight_matrix(raw_embedding, vocab)

print("Vocab size: ", vocab_size)
print("Max text length: ", max_length)
print("Embedding dim: ", embedding_dim)

BATCH_SIZE = 1024
EPOCHS = 15

# create the embedding layer
embedding_layer = Embedding(vocab_size, 
                            embedding_dim, 
                            weights = [embedding_matrix], 
                            input_length = max_length, 
                            trainable = False)

# define model
model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.2))
model.add(LSTM(200, dropout = 0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation = "sigmoid"))

print(model.summary())

model.compile(optimizer = "adam", loss = 'binary_crossentropy', metrics = ['accuracy'])

# callbacks
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', 
                              factor = 0.1,
                              min_lr = 0.01)

# train model
history = model.fit(x_train, y_train, batch_size = BATCH_SIZE, epochs = EPOCHS,
                    validation_split = 0.1, verbose = 1, callbacks = [reduce_lr])

# evaluate model
score = model.evaluate(x_test, y_test, batch_size = BATCH_SIZE)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# save model
model.save('model_final.h5')

# plotting model performance
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss'] 
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label = 'Training acc')
plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()