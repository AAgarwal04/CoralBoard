import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU
import numpy as np
import pandas as pd

def load_data(file_path, sheet_name, skip_rows=0, nrows=None):
    if nrows is None:
        df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=skip_rows)
    else:
        df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=skip_rows, nrows=nrows)
    return df

def preprocess_data(df, for_training=True, nrows=None):
    feature_columns = ['RH', 'Light', 
                    #    'Pressure',
                        'WifiAmnt', 'WifiAvg', 'WifiMax', 'BLEAmnt', 'BLEAvg', 'BLEMax']
    available_columns = df.columns.intersection(feature_columns)
    
    if nrows is not None:
        X = df[available_columns].iloc[:nrows].to_numpy()
    else:
        X = df[available_columns].to_numpy()
    
    if for_training:
        if nrows is not None:
            y = df['Inside'].iloc[:nrows].to_numpy()
        else:
            y = df['Inside'].to_numpy()
        return X, y
    else:
        return X

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(8,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model