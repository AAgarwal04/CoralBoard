import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

converter = tf.lite.TFLiteConverter.from_keras_model(keras.models.load_model("environmentModel.h5"))
tflite_model = converter.convert()

# Save the model.
with open('environmentModel.tflite', 'wb') as f:
  f.write(tflite_model)

# Load the Excel file
excel_path = 'Data/Data.xlsx'
df = pd.read_excel(excel_path)

# Select the 9 columns you want to use
feature_columns = ['RH', 'Light', 'Pressure', 'WifiAmnt', 'WifiAvg', 'WifiMax', 'BLEAmnt', 'BLEAvg', 'BLEMax']
df_selected = df[feature_columns]

# Normalize the data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(df_selected).astype(np.float32)  # Convert to float32

# Function to get random batches
def get_random_batch(data, batch_size=1):
    indices = np.random.randint(0, data.shape[0], size=batch_size)
    return data[indices]

# Representative data generator
def representative_data_gen():
    for _ in range(100):  # Generate 100 sample inputs
        sample = get_random_batch(normalized_data)
        yield [sample]

# Load your model
model = tf.keras.models.load_model('environmentModel.h5')

# Create TFLite converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Set optimization flag
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Set representative dataset
converter.representative_dataset = representative_data_gen

# Optionally, enforce full integer quantization
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Convert the model
tflite_quant_model = converter.convert()

# Save the quantized model
with open('environmentModel_quantized.tflite', 'wb') as f:
    f.write(tflite_quant_model)

print("Quantized model saved successfully.")