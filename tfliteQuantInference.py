from utils import load_data, preprocess_data
import tensorflow as tf
import joblib
import numpy as np

# Load the saved TFLite model
interpreter = tf.lite.Interpreter(model_path='environmentModel_quantized.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load the scaler
scaler = joblib.load('scaler.pkl')

df = load_data('Data/Data.xlsx', sheet_name='testOutside')
X = preprocess_data(df, for_training=False)
# Load and preprocess test data
# Assuming you have a function to load data into a numpy array
# X = np.array([
#     [53, 499, 171, 48, 87, 65, -81, -57],
#     [65, 31000, 28, 50, 80, 40, -80, -65],
#     [55, 30000, 24, 47, 75, 38, -83, -67]
# ])

# Define a prediction function
def predict(X):
    X_scaled = scaler.transform(X)
    results = []
    for row in X_scaled:
        # Check if the model expects quantized input
        if input_details[0]['dtype'] == np.uint8 or input_details[0]['dtype'] == np.int8:
            input_scale, input_zero_point = input_details[0]['quantization']
            row_quantized = row / input_scale + input_zero_point
            row_quantized = row_quantized.astype(input_details[0]['dtype'])
            interpreter.set_tensor(input_details[0]['index'], np.expand_dims(row_quantized, axis=0))
        else:
            interpreter.set_tensor(input_details[0]['index'], np.expand_dims(row, axis=0).astype(np.float32))
        
        interpreter.invoke()
        
        output = interpreter.get_tensor(output_details[0]['index'])
        
        # Dequantize the output if necessary
        if output_details[0]['dtype'] == np.uint8 or output_details[0]['dtype'] == np.int8:
            output_scale, output_zero_point = output_details[0]['quantization']
            output = (output.astype(np.float32) - output_zero_point) * output_scale
        
        results.append(output[0])
    return np.array(results)

# Perform inference
start = 0
end = min(start + 50, len(X))
X_subset = X[start:end]

predictions = predict(X_subset)

features = ["RH", "Light", 
            # "Pressure", 
            "WifiAmnt", "WifiAvg", "WifiMax", "BLEAmnt", "BLEAvg", "BLEMax"]

# Print results
for i, (row, prediction) in enumerate(zip(X_subset, predictions)):
    print(f"Row {i+start+1}:")
    print(f"  Raw Data:")
    for feature, value in zip(features, row):
        print(f"    {feature}: {value}")
    print(f"  Prediction: {'Inside' if prediction > 0.5 else 'Outside'}")
    print(f"  Probability: {prediction[0]:.4f}")
    print()

print(f"Total predictions made: {len(predictions)}")