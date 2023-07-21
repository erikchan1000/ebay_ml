import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping
import csv
import os

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
from transformers import BertTokenizerFast
from transformers import TFBertModel

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

