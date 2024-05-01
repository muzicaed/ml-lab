import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)


def fetch_datasest():
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                    'Acceleration', 'Model Year', 'Origin']

    raw_dataset = pd.read_csv(url, names=column_names,
                              na_values='?', comment='\t',
                              sep=' ', skipinitialspace=True)

    dataset = raw_dataset.copy()
    dataset = dataset.dropna()
    dataset['Origin'] = dataset['Origin'].map(
        {1: 'USA', 2: 'Europe', 3: 'Japan'})
    dataset = pd.get_dummies(
        dataset, columns=['Origin'], prefix='', prefix_sep='')
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
    return (train_dataset, test_dataset)


def create_normalizer(train_features):
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(train_features))
    normalizer.mean.numpy()
    return normalizer


def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))
    return model


def plot_predict(predictions):
    plt.axes(aspect='equal')
    plt.scatter(test_labels, predictions)
    plt.xlabel('True Values [Horsepower]')
    plt.ylabel('Predictions [Horsepower]')
    lims = [30, 100]
    plt.xlim(lims)
    plt.ylim(lims)
    plt.plot(lims, lims)
    plt.show()


(train_dataset, test_dataset) = fetch_datasest()
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('Horsepower')
test_labels = test_features.pop('Horsepower')

dnn_model = build_and_compile_model(create_normalizer(train_features))
dnn_model.summary()
history = dnn_model.fit(
    train_features,
    train_labels,
    validation_split=0.2,
    verbose=0, epochs=100)

result = dnn_model.evaluate(test_features, test_labels)

predictions = dnn_model.predict(test_features).flatten()
plot_predict(predictions)

dnn_model.save('dnn_model.keras')
