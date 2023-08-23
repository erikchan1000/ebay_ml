import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from keras.utils import pad_sequences
from transformers import BertTokenizer

class ProcessData: 

    def __init__(self, df):
        self.df = df
        self.unique_tags = self.__get_unique_tags()
        self.tag2id = self.__get_tag2id()
        self.id2tag = self.__get_id2tag()
        self.max_length = 61
        self.tz = BertTokenizer.from_pretrained('bert-base-cased')

    def getmap(self):
        return self.token2id
        
    def __get_unique_tags(self):
        unique_tags = self.df['Tag'].explode().unique()
        return unique_tags
        
    def get_id2tag(self):
        return self.id2tag
    
    def __get_tag2id(self):
        tag2id = {tag: i for i, tag in enumerate(self.unique_tags)}
        
        return tag2id
    
    def __get_id2tag(self):
        id2tag = {v: k for k, v in self.tag2id.items()}
        #add padding to id2tag
        id2tag[len(id2tag.keys())] = '__PADDING'
        return id2tag
    

    
    def convert_tokens_tags_to_ids(self):
        new_df = self.df.copy()
        new_df['Token'] = new_df['Token'] = new_df['Token'].apply(lambda x: self.tz.convert_tokens_to_ids(x))

        print(new_df['Token'].head(50))
        new_df['Tag'] = new_df['Tag'].apply(lambda x: [self.tag2id[tag] for tag in x])
        return new_df
    
    def pad_sequences(self):
        x = self.convert_tokens_tags_to_ids()['Token'].tolist()
        y = self.convert_tokens_tags_to_ids()['Tag'].tolist()
        max_x_id = max([max(x) for x in x])
        max_y_id = max([max(y) for y in y])
        #print(max_y_id)
        x = pad_sequences(x, padding='post', truncating='post', value=max_x_id+1 , maxlen=self.max_length)
        y = pad_sequences(y, padding='post', truncating='post', value=max_y_id+1, maxlen=self.max_length)

        return x, y, max_x_id, max_y_id

    def reformat_output(self, output):
        #output has shape (inputs, max_length, num_tags)

        #convert output of logits to tag ids
        output = np.argmax(output, axis=-1)

        #convert tag ids to tags
        output = [[self.id2tag[tag_id] for tag_id in row] for row in output]
        #remove padding
        output = [[tag for tag in row if tag != '__PADDING'] for row in output]

        return output
    

class ProcessQuiz:
    def __init__(self, df, id2tag):
        self.df = df
        self.max_length = 61
        self.id2tag = id2tag

    
    def convert_tokens_to_ids(self):
        new_df = self.df.copy()
        new_df['Title'] = new_df['Title'] = new_df['Title'].apply(lambda x: self.tz.convert_tokens_to_ids(x))
        return new_df
    
    def pad_title(self):
        x = self.convert_tokens_to_ids()['Title'].tolist()
        max_x_id = max([max(x) for x in x])
        x = pad_sequences(x, padding='post', truncating='post', value=max_x_id+1, maxlen=self.max_length)
        return x, max_x_id
    
    def reformat_output(self, output):
        #convert output of logits to tag ids
        output = np.argmax(output, axis=-1)

        #convert tag ids to tags
        output = [[self.id2tag[tag_id] for tag_id in row] for row in output]
        #remove padding
        output = [[tag for tag in row if tag != '__PADDING'] for row in output]

        return output
    
    def reprocess_data(self, reformatted_output):
        #reformatted_output is a list of lists of tags
        #create a new column in df called Tag
        self.df['Tag'] = reformatted_output
        print(self.df.head(50))
        #separate each title token to it's own row with the corresponding tag
        #explode both Title and Tag columns
        print(self.df.head(50))
        self.df = self.df.explode(['Title', 'Tag'])

        return self.df

