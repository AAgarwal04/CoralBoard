import numpy as np
from pycoral.utils import edgetpu
from pycoral.adapters import common
from pycoral.adapters import classify

# Load the saved EdgeTPU model
model_path = 'environmentModel_edgetpu.tflite'
interpreter = edgetpu.make_interpreter(model_path)
interpreter.allocate_tensors()

# Get input details
input_details = interpreter.get_input_details()
input_scale, input_zero_point = input_details[0]['quantization']

X = np.array([
    [53, 499, 101, 171, 48.73099, 87, 65, -81.30676, -57],
    [65, 31000, 102, 28, 50.0, 80, 40, -80.0, -65],
    [55, 30000, 100, 24, 47.0, 75, 38, -83.0, -67]
])

# Uncomment and modify the following line to input your own array for inference
# X = np.array([[your_data_here]])

# Define a prediction function
def predict(X):
    results = []
    for row in X:
        # Quantize the input
        quantized_input = np.uint8(row / input_scale + input_zero_point)
        common.set_input(interpreter, quantized_input)
        
        # Run inference
        interpreter.invoke()
        
        # Get the output
        output = classify.get_scores(interpreter)
        results.append(output[0])
    return np.array(results)

# Perform inference
predictions = predict(X)

features = ["RH", "Light", "Pressure", "WifiAmnt", "WifiAvg", "WifiMax", "BLEAmnt", "BLEAvg", "BLEMax"]

# Print results
for i, (row, prediction) in enumerate(zip(X, predictions)):
    print(f"Row {i+1}:")
    print(f"  Raw Data:")
    for feature, value in zip(features, row):
        print(f"    {feature}: {value}")
    print(f"  Prediction: {'Inside' if prediction > 0.5 else 'Outside'}")
    print(f"  Probability: {prediction:.4f}")
    print()

print(f"Total predictions made: {len(predictions)}")