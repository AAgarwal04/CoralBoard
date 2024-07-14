from utils import load_data, preprocess_data
import tensorflow as tf
import joblib
import numpy as np
import shap
import random

# Load the saved model and scaler
model = tf.keras.models.load_model('environmentModel.h5')
scaler = joblib.load('scaler.pkl')

# Load and preprocess test data
df = load_data('Data/Data.xlsx', sheet_name='testOutside')
X = preprocess_data(df, for_training=False)

# Create a SHAP explainer
explainer = shap.KernelExplainer(model.predict, scaler.transform(X))

# Perform inference on each row
results = []
start = 0
index = start

randList = []
for i in range(0, 20):
    n = random.randint(0, len(X))
    randList.append(n)
print(randList)

for ind in randList:
# while (index-start) < 50:
# while index < len(X):
    row = X[index]
    # row = np.array([[60, 30504, 101, 26, 48.57692308, 77, 39, -81.71794872, -66]], dtype=np.float32)
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