import pandas as pd
import numpy as np
import csv
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences



class CleanData:
    def __init__(self, train_file, eval_file):
        self.df_train= train_file
        self.df_eval_base = eval_file
        self.unique_tokens = self.df_eval_base['Tag'].unique()
        self.df_eval = self.df_eval_base.drop(columns=['Token']).groupby(['Record Number', 'Title'])['Tag'].apply(list).reset_index()
        self.x = self.df_eval['Title']
        self.y = self.df_eval['Tag']
        self.maxLen = 0
        self.trainLen = len(self.df_train)
        self.evalLen = len(self.df_eval)
        self.x_preprocessed = self.process_x(x)

        
        for title in self.x:
            if len(title) > self.maxLen:
                self.maxLen = len(title)

    def word2id(self):
      tokenizer = Tokenizer()
      tokenizer.fit_on_texts(self.x)
      
      word2id = tokenizer.word_index
      return word2id
    
    def id2word(self, word_index):
      return {v: k for k, v in word_index.items()}
    
    def tags2id(self):
      tags2id = {}
      for i, tag in enumerate(self.unique_tokens):
          tags2id[tag] = i
      return tags2id
    
    def id2tags(self, tags2id):
      return {tag: i for tag, i in tags2id.items()}
    
    def process_x(self, x):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.x)
        sequences = tokenizer.texts_to_sequences(x)
        return pad_sequences(sequences, maxlen=self.maxLen, padding='post', value=37)


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
