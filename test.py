import numpy as np
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters.common import input_details, output_details, set_input_tensor

# Load the pre-quantized TFLite model
interpreter = make_interpreter('environmentModel_quantized.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = input_details(interpreter)
output_details = output_details(interpreter)

# Define your input data (assuming it's already in the correct format)
X = np.array([
    [55, 1293, 159, 47, 92, 94, -78, -40]
], dtype=np.int8)  # Assuming int8 quantization

def predict(X):
    # Set the input tensor
    set_input_tensor(interpreter, X)

    # Run inference
    interpreter.invoke()

    # Get the output
    output = interpreter.get_tensor(output_details['index'])

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
