import numpy as np
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters.common import input_details, output_details, set_input_tensor

# Load the Edge TPU compiled model
interpreter = make_interpreter('environmentModel_edgetpu.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = input_details(interpreter)
output_details = output_details(interpreter)

# Define your input data
# This should be scaled and quantized to match what the model expects
X = np.array([
    [55, 127, 100, 47, 92, 94, -78, -40]
], dtype=np.int8)  # Ensure this matches your model's expected input

def predict(X):
    # Set the input tensor
    set_input_tensor(interpreter, X)

    # Run inference
    interpreter.invoke()

    # Get the output
    output = interpreter.get_tensor(output_details['index'])

    # If the output is quantized, you might need to dequantize it
    if output_details['quantization'] != (0.0, 0):
        scale, zero_point = output_details['quantization']
        output = (output.astype(np.float32) - zero_point) * scale

    return output

# Perform inference
predictions = predict(X)

# Print results
features = ["RH", "Light", "WifiAmnt", "WifiAvg", "WifiMax", "BLEAmnt", "BLEAvg", "BLEMax"]
for i, (row, prediction) in enumerate(zip(X, predictions)):
    print(f"Row {i+1}:")
    print(f"  Raw Data:")
    for feature, value in zip(features, row):
        print(f"    {feature}: {value}")
    print(f"  Prediction: {'Inside' if prediction > 0.5 else 'Outside'}")
    print(f"  Probability: {prediction[0]:.4f}")
    print()

print(f"Total predictions made: {len(predictions)}")
