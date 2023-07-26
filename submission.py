import pandas as pd
import numpy as np
import tensorflow as tf
import os
from clean_data import CleanData
from process_data import ProcessQuiz
from process_data import ProcessData
from transformers import TFBertModel

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



from clean_data import CleanData

test_data = CleanData('data/Train_Tagged_Titles.tsv').clean_data()
processed_test = ProcessData(test_data)
id2tag = processed_test.get_id2tag()
token2id = processed_test.get_token2id()

x, y, max_x_id, max_y_id = processed_test.pad_sequences()
print(max_x_id, max_y_id)
print(id2tag)

test = CleanData('data/Listing_Titles.tsv')

quiz_data = test.get_quiz_data()

#get max length of title 
#title column is already an array of tokens



tf.keras.utils.get_custom_objects()['TFBertModel'] = TFBertModel
model = tf.keras.models.load_model('./models/bert_model.h5')

processed_data = ProcessQuiz(quiz_data, token2id, id2tag)

def get_max_length():
  quiz_x, max_x_id = processed_data.pad_title()
  print(quiz_x.shape)

  all_data = test.get_all_data()
  print('max_length')
  print(all_data['Title'].apply(lambda x: len(x)).max())

x, max_x = processed_data.pad_title()
print(x)
print(x.shape)

pred = model.predict(x[:50])
print(pred.shape)

reformat = processed_data.reformat_output(pred)
print(quiz_data[:50])
print(reformat[:50])