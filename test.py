import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping
import os
from clean_data import CleanData
from keras.utils import pad_sequences
from process_data import ProcessData

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Device:', tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()
print('Number of replicas:', strategy.num_replicas_in_sync)

import transformers
from transformers import TFBertModel
from transformers import AutoTokenizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

train_df = CleanData('data/Train_Tagged_Titles.tsv').clean_data()


processed_df = ProcessData(train_df)
train_x, train_y, max_x_id, max_y_id = processed_df.pad_sequences()


print(train_x.shape)
print(len(train_x))
val_x, val_y = train_x[4500:], train_y[4500:] 

tf.keras.utils.get_custom_objects()['TFBertModel'] = TFBertModel
#load model from models/bert_model
model = tf.keras.models.load_model('./models/bert_model.h5')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

print(val_x.shape, val_y.shape)

pred = model.predict(val_x)
print(pred.shape)

reformat = processed_df.reformat_output(pred)
print(reformat[0])
print(val_y[0])
print(processed_df.id2tag)

#print processed_df with all columns and first 50 rows
pd.set_option('display.max_columns', None)
print(processed_df.df[-500 : ])