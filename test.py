from utils import load_data, preprocess_data
import tensorflow as tf
import joblib
import numpy as np
import shap

# Load the saved model and scaler
model = tf.keras.models.load_model('environmentModel.h5')
scaler = joblib.load('scaler.pkl')

# Load and preprocess test data
df = load_data('/home/aryan/Documents/Coral/Data.xlsx', sheet_name='kidOutside')
X = preprocess_data(df, for_training=False)

# Create a SHAP explainer
explainer = shap.KernelExplainer(model.predict, scaler.transform(X))

# Perform inference on each row
results = []
start = 50
index = start
while (index - start) < 10:
    row = X[index]
    row_reshaped = row.reshape(1, -1)
    row_scaled = scaler.transform(row_reshaped)
    prediction = model.predict(row_scaled)
    binary_prediction = (prediction > 0.5).astype(int)
    shap_values = explainer.shap_values(row_scaled)
    results.append({
        'raw_data': row,
        'probability': prediction[0][0],
        'prediction': 'Inside' if binary_prediction[0][0] == 1 else 'Outside',
        'shap_values': shap_values[0]
    })
    index += 1

features = ["RH", "Light",	"Pressure",	"WifiAmnt",	"WifiAvg", "WifiMax", "BLEAmnt", "BLEAvg", "BLEMax"]


# Print results
for i, result in enumerate(results):
    print(f"Row {i+1}:")
    print(f"  Prediction: {result['prediction']}")
    print(f"  Probability: {result['probability']:.4f}")
    print("  SHAP Values:")
    shap_values = result['shap_values']
    for j, value in enumerate(shap_values):
        print(f"    {features[j]}: {value}")
    print()