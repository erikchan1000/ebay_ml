import pandas as pd
import numpy as np
import csv
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences



class CleanData:
    def __init__(self, path):
        self.path = path

    def clean_data(self):
        df_eval_base = pd.read_csv(self.path, sep='\t', dtype=str, keep_default_na=False, na_values=[""], quoting=csv.QUOTE_NONE)
        print(df_eval_base.head())

        print(f"Unique tags: {df_eval_base.Tag.unique()}")

        df_eval = df_eval_base.drop(columns=['Title'])

        print(df_eval.head(50))

        #use record number as the new index
        #create a new dataframe with the record number as the index, tokens as list of tokens, and tags as list of tags

        df_tokens = df_eval.groupby(['Record Number'])['Token'].apply(list)

        #reset record number to start from 1 and increment by 1
        df_tokens = df_tokens.reset_index()
        df_tokens['Record Number'] = df_tokens.index + 1
        print(df_tokens.head(50))

        #create new data frame with list of tags
        df_tags = df_eval.groupby(['Record Number'])['Tag'].apply(list)
        df_tags = df_tags.reset_index()
        df_tags['Record Number'] = df_tags.index + 1
        print(df_tags.head(50))

        #merge the two dataframes
        df = pd.merge(df_tokens, df_tags, on='Record Number')

        return df

