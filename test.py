from utils import load_data, preprocess_data
import tensorflow as tf
import joblib
import numpy as np

# Load the saved model and scaler
model = tf.keras.models.load_model('trained_model.h5')
scaler = joblib.load('scaler.pkl')

# Load and preprocess test data
df = load_data('/home/aryan/Documents/Coral/Data.xlsx', sheet_name='queenOutside')
X = preprocess_data(df, for_training=False)
# Perform inference on each row
results = []
start = 49
index = start
while (index - start) < 100:
    row = X[index]
    row_reshaped = row.reshape(1, -1)
    row_scaled = scaler.transform(row_reshaped)
    prediction = model.predict(row_scaled)
    binary_prediction = (prediction > 0.5).astype(int)
    results.append({
        'raw_data': row,
        'probability': prediction[0][0],
        'prediction': 'Inside' if binary_prediction[0][0] == 1 else 'Outside'
    })
    index += 1

# Print results
for i, result in enumerate(results):
    print(f"Row {i+1}:")
    print(f"  Prediction: {result['prediction']}")
    print(f"  Probability: {result['probability']:.4f}")
    print()