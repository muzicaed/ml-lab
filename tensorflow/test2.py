import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses

batch_size = 32
raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    '/Users/mikaelhellman/dev/ml-lab/aclImdb/test',
    batch_size=batch_size)


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')


model = tf.keras.models.load_model('test.keras', custom_objects={
                                   "custom_standardization": custom_standardization})

examples = tf.constant([
    "The movie was great!",
    "The movie was okay.",
    "The movie was terrible..."
])

predictions = model.predict(examples)

print(np.argmax(predictions[2]))
