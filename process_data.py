import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from keras.utils import pad_sequences

class ProcessData: 
    #Erik should hard code the map

    def __init__(self, df):
        self.df = df
        self.unique_tokens = self.__get_unique_tokens()
        self.unique_tags = self.__get_unique_tags()
        self.token2id = self.__get_token2id()
        self.id2token = self.__get_id2token()
        self.tag2id = self.__get_tag2id()
        self.id2tag = self.__get_id2tag()

    def getmap(self):
        return self.token2id

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
        #print(id2tag)
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
        #print(max_x_id)
        max_y_id = max([max(y) for y in y])
        #print(max_y_id)
        x = pad_sequences(x, padding='post', truncating='post', value=max_x_id+1)
        y = pad_sequences(y, padding='post', truncating='post', value=max_y_id+1)

        return x, y, max_x_id, max_y_id

    def tokenize_title(self, title_list):
        return title_list.split(" ")

    def reformat_output(self, output):
        #map for id -> tag
        map = self.id2tag
        map[len(map.keys())] = 'PADDING'
        #df should be padded/cleaned already 
        
        tokens = pd.DataFrame(columns = ['Record', 'Token'])
        for x in range(self.df.shape[0]):
            for token in self.df['Token'][x]:
                add = pd.series(data = [x+1, token], index = ['Record', 'Token'])
                tokens.append(add, ignore_index = True)
        

        #model output is sequential list of tag ids that match up w the order of tokens
        #need to assign record numbers / tokens back to the tag ids
        
        tokens['Tag'] = output

        for x in tokens['Tag']:
            name = map[x]
            tokens.replace['Tag'][x] = name

        print(tokens)

        #tokens is df of [record number, tokens, tag_id]

        '''
        final = pd.DataFrame(columns = [''])
        curr = []
        for x in range(len(output)):
            #need to merge consecutive tokens that have same tags (NaN)
            if output[x] == 37 and len(curr) == 0:
                curr.append(x)
                curr.append(tokens['Token'][x])

            elif output[x] == 37:
                curr.append(tokens['Token'][x])

            elif len(curr) > 0 and output[x] != 37:

                curr = []

            else:
                curr = []


        #output should be tab seperated values: [record number] [aspect name (tag)] [aspect value (token)]
        '''

    

    
