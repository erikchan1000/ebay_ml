import pandas as pd
import numpy as np
import csv
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences



class CleanData:
    def __init__(self, path):
        self.path = path

    def test(self):
        
        df = pd.read_csv(self.path, sep='\t', dtype=str, keep_default_na=False, na_values=[""], quoting=csv.QUOTE_NONE)
        #choose 2500 random rows

        return df

    def get_quiz_data(self):
        
        df = pd.read_csv(self.path, sep='\t', dtype=str, keep_default_na=False, na_values=[""], quoting=csv.QUOTE_NONE)
        #choose 2500 random rows
        df = df.sample(n=2500, random_state=1)
        #consists of columns record number and title
        #create new data frame with list of title separated by space
        df['Title'] = df['Title'].str.split()
        
        return df
    
    def get_all_data(self):
        df = pd.read_csv(self.path, sep='\t', dtype=str, keep_default_na=False, na_values=[""], quoting=csv.QUOTE_NONE)
        #choose 2500 random rows
        #consists of columns record number and title
        #create new data frame with list of title separated by space
        df['Title'] = df['Title'].str.split()
        
        return df

    def clean_data(self):
        df_eval_base = pd.read_csv(self.path, sep='\t', dtype=str, keep_default_na=False, na_values=[""], quoting=csv.QUOTE_NONE)

        df_eval = df_eval_base.drop(columns=['Title'])


        #use record number as the new index
        #new df_tokens data frame that merge all tokens with same record number into a list of tokens while maintaining the order
        df_tokens = df_eval.groupby(['Record Number'])['Token'].apply(list)

        #create new data frame with list of tags
        df_tags = df_eval.groupby(['Record Number'])['Tag'].apply(list)

        #merge the two dataframes
        df = pd.merge(df_tokens, df_tags, on='Record Number')

        #sort the data frame by record number
        df = df.sort_values(by=['Record Number'])
        
        #assign Record Number key to column index
        df['Record Number'] = df.index
        #reset index
        df = df.reset_index(drop=True, inplace=False)
        #move Record Number column to the front
        cols = df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        df = df[cols]

        df['Record Number'] = df['Record Number'].astype(int)
        df = df.sort_values(by=['Record Number'])

        df = df.reset_index(drop=True, inplace=False)
        return df

