import pandas as pd
import numpy as np
import csv 
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from collections import defaultdict
import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, Bidirectional

df_train = pd.read_csv('./data/Listing_Titles.tsv', sep='\t', dtype=str, keep_default_na=False, na_values=[""], quoting=csv.QUOTE_NONE)
df_eval_base = pd.read_csv('./data/Train_Tagged_Titles.tsv', sep='\t', dtype=str, keep_default_na=False, na_values=[""], quoting=csv.QUOTE_NONE)

#Train has four columns: Record Number, Title, Token, Tag
#Train data matches records 1 to 5000 of listing data, inclusively
#Quiz Data consists of records 5001 to 30000 of the listing data inclusively

#NaN tags indicates that the token are part of the previous token tag
df_train = df_train[0:5000]

print(df_train.head())
print(df_eval_base.head())

print(df_eval_base['Tag'].unique())

#input consists of title
#must separate each word in title into a token
#then must categorize each token into a tag
unique_tokens = df_eval_base['Tag'].unique()
df_eval = df_eval_base.drop(columns=['Token'])
df_eval = df_eval.groupby(['Record Number', 'Title'])['Tag'].apply(list).reset_index()
print(df_eval.head())

x = df_eval['Title']
y = df_eval['Tag']

print(x.head())
print(y.head())



maxLen = 1100
trainLen = 2500
evalLen = 2500

#Tokenize the data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x)
sequences = tokenizer.texts_to_sequences(x)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
ind2word = {v: k for k, v in word_index.items()}


word2id = word_index
id2word = {}
for word, i in word2id.items():
    id2word[i] = word

x_preprocessed = pad_sequences(sequences, maxlen=maxLen, padding='post')

print(df_eval['Tag'])
tags2id = {}
for i, tag in enumerate(unique_tokens):
    tags2id[tag] = i

id2tags = {}
for tag, i in tags2id.items():
    id2tags[i] = tag

max_len_x = 0
for x in x_preprocessed:
    if len(x) > max_len_x:
        max_len_x = len(x)



def preprocess_tags(tags2id, y_ready):
    y_preprocessed = []
    for y in y_ready:
        y_temp = []
        for tag in y:
            y_temp.append(tags2id[tag])


        y_preprocessed.append(y_temp)

    return pad_sequences(y_preprocessed, maxlen=max_len_x, padding='post', value=0)


y_preprocessed = preprocess_tags(tags2id, y)

x_train = x_preprocessed[0:trainLen]
y_train = y_preprocessed[0:trainLen]

x_val = x_preprocessed[trainLen:trainLen+evalLen]
y_val = y_preprocessed[trainLen:trainLen+evalLen]


train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
eval_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
train_dataset = train_dataset.batch(32)
eval_dataset = eval_dataset.batch(32)

print(f'x_train shape: {x_train.shape}')
print(f'y_train shape: {y_train.shape}')

print(f'x_val shape: {x_val.shape}')
print(f'y_val shape: {y_val.shape}')

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(word_index) + 1, 128, input_length=maxLen),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(len(unique_tokens), activation='softmax'))
])

print(model.summary())

model.compile(loss='sparse_categorical_crossentropy',
                optimizer=tf.keras.optimizers.Adam(1e-4),
                metrics=['accuracy'])

history = model.fit(train_dataset, epochs=10, validation_data=eval_dataset)
