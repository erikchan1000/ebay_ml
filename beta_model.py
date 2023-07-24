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
print(train_df.head(50))

#BERT Training model

processed_df = ProcessData(train_df)
train_x, train_y, max_x_id, max_y_id = processed_df.pad_sequences()

print(train_x.shape)
print(train_y.shape)

train_x = train_x[:4501]
train_y = train_y[:4501]
eval_x = train_x[4501:]
eval_y = train_y[4501:]

#split data into train and validation sets
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

print(train_x.shape)
print(train_y.shape)
print(val_x.shape)
print(val_y.shape)


#build model
def build_model(max_x_id, max_y_id):
    input_word_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input_word_ids')
    bert_layer = TFBertModel.from_pretrained('bert-base-uncased')
    bert_outputs = bert_layer(input_word_ids)
    last_hidden_state = bert_outputs[0]
    print(last_hidden_state)
    output = tf.keras.layers.Dense(max_y_id+2, activation='softmax')(last_hidden_state)
    model = tf.keras.Model(inputs=input_word_ids, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    return model

with strategy.scope():
    model = build_model(max_x_id, max_y_id)
    model.summary()

#train model
early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='min', restore_best_weights=True)
history = model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=15, batch_size=32, callbacks=[early_stopping], verbose=2)

#plot accuracy and loss
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel('Epochs')
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()

plot_graphs(history, 'accuracy')

plot_graphs(history, 'loss')

#save model
