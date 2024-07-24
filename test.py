from utils import load_data, preprocess_data
import joblib
import numpy as np
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters.common import input_size, set_input
from pycoral.adapters.classify import get_classes

# Load the saved TFLite model
interpreter = make_interpreter('environmentModel_quantized.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load the scaler
scaler = joblib.load('scaler.pkl')

# Define your NumPy array
X = np.array([
    [55, 1293, 159, 46.874, 92, 94, -78.095, -40]
])

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
            set_input(interpreter, row_quantized)
        else:
            set_input(interpreter, row.astype(np.float32))
        
        interpreter.invoke()
        
        output = interpreter.get_tensor(output_details[0]['index'])
        
        # Dequantize the output if necessary
        if output_details[0]['dtype'] == np.uint8 or output_details[0]['dtype'] == np.int8:
            output_scale, output_zero_point = output_details[0]['quantization']
            output = (output.astype(np.float32) - output_zero_point) * output_scale
        
        results.append(output[0])
    return np.array(results)

# Perform inference
predictions = predict(X)

features = ["RH", "Light", "WifiAmnt", "WifiAvg", "WifiMax", "BLEAmnt", "BLEAvg", "BLEMax"]

# Print results
for i, (row, prediction) in enumerate(zip(X, predictions)):
    print(f"Row {i+1}:")
    print(f"  Raw Data:")
    for feature, value in zip(features, row):
        print(f"    {feature}: {value}")
    print(f"  Prediction: {'Inside' if prediction > 0.5 else 'Outside'}")
    print(f"  Probability: {prediction[0]:.4f}")
    print()

print(f"Total predictions made: {len(predictions)}")
