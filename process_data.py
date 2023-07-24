import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from keras.utils import pad_sequences

class ProcessData: 
    def __init__(self, df):
        self.df = df
        self.unique_tokens = self.__get_unique_tokens()
        self.unique_tags = self.__get_unique_tags()
        self.token2id = self.__get_token2id()
        self.id2token = self.__get_id2token()
        self.tag2id = self.__get_tag2id()
        self.id2tag = self.__get_id2tag()


    def __get_unique_tokens(self):
        unique_tokens = self.df['Token'].explode().unique()
        return unique_tokens
    
    def __get_unique_tags(self):
        unique_tags = self.df['Tag'].explode().unique()
        return unique_tags
    
    def __get_token2id(self):
        token2id = {token: i for i, token in enumerate(self.unique_tokens)}
        return token2id
    
    def __get_id2token(self):
        id2token = {v: k for k, v in self.token2id.items()}
        return id2token
    
    def __get_tag2id(self):
        tag2id = {tag: i for i, tag in enumerate(self.unique_tags)}
        return tag2id
    
    def __get_id2tag(self):
        id2tag = {v: k for k, v in self.tag2id.items()}
        return id2tag
    
    def convert_tokens_tags_to_ids(self):
        new_df = self.df.copy()
        new_df['Token'] = new_df['Token'].apply(lambda x: [self.token2id[token] for token in x])
        new_df['Tag'] = new_df['Tag'].apply(lambda x: [self.tag2id[tag] for tag in x])
        return new_df

    
    def pad_sequences(self):
        x = self.convert_tokens_tags_to_ids()['Token'].tolist()
        y = self.convert_tokens_tags_to_ids()['Tag'].tolist()
        max_x_id = max([max(x) for x in x])
        max_y_id = max([max(y) for y in y])
        x = pad_sequences(x, padding='post', truncating='post', value=max_x_id+1)
        y = pad_sequences(y, padding='post', truncating='post', value=max_y_id+1)

        return x, y, max_x_id, max_y_id
    

    
