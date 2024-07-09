import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd

#inputs
df = pd.read_excel('/Users/Aryan/Documents/Coral/Data.xlsx', sheet_name="Combined")
rh = df['RH'].to_numpy()
light = df['Light'].to_numpy()
pressure = df['Pressure'].to_numpy()
uv = df['UV'].to_numpy()
wifiAmnt = df['WifiAmnt'].to_numpy()
wifiAvg = df['WifiAvg'].to_numpy()
wifiMax = df['WifiMax'].to_numpy()
bleAmnt = df['BLEAmnt'].to_numpy()
bleAvg = df['BLEAvg'].to_numpy()
bleMax = df['BLEMax'].to_numpy()

y = df['Inside'].to_numpy()

X = np.column_stack((rh, light, pressure, uv, wifiAmnt, wifiAvg, wifiMax, bleAmnt, bleAvg, bleMax))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Normalize the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training data shape:", X_train_scaled.shape)
print("Training labels shape:", y_train.shape)
print("Testing data shape:", X_test_scaled.shape)
print("Testing labels shape:", y_test.shape)

print("\nFirst few samples of scaled training data:")
print(X_train_scaled[:5])
print("\nFirst few training labels:")
print(y_train[:5])

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

train_dataset = tf.data.Dataset.from_tensor_slices((X_train_scaled, y_train)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test_scaled, y_test)).batch(32)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_dataset,
    epochs=100,
    validation_data=test_dataset,
    class_weight=class_weight_dict
)

test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test accuracy: {test_accuracy:.4f}")

