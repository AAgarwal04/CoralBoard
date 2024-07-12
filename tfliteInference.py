import numpy as np
from pycoral.utils import edgetpu
from pycoral.adapters import common
from pycoral.adapters import classify

# Load the saved EdgeTPU model
model_path = 'environmentModel_quantized.tflite'
interpreter = edgetpu.make_interpreter(model_path)
interpreter.allocate_tensors()

# Get input details
input_details = interpreter.get_input_details()
input_shape = input_details[0]['shape']
print("Expected input shape:", input_shape)

X = np.array([
    [53, 499, 101, 171, 48, 87, 65, -81, -57],
    [65, 31000, 102, 28, 50, 80, 40, -80, -65],
    [55, 30000, 100, 24, 47, 75, 38, -83, -67]
])

def predict(X):
    results = []
    for row in X:
        # Convert to float32 and reshape
        input_data = row.astype(np.float32).reshape(1, 9)
        
        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        interpreter.invoke()
        
        # Get the output
        output_details = interpreter.get_output_details()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        results.append(output_data.flatten()[0])  # Flatten and take the first element
    return np.array(results)

# Perform inference
predictions = predict(X)

features = ["RH", "Light", "Pressure", "WifiAmnt", "WifiAvg", "WifiMax", "BLEAmnt", "BLEAvg", "BLEMax"]

# Print results
for i, (row, prediction) in enumerate(zip(X, predictions)):
    print("Row {}:".format(i+1))
    print("  Raw Data:")
    for feature, value in zip(features, row):
        print("    {}: {}".format(feature, value))
    print("  Prediction: {}".format('Inside' if prediction > 0.5 else 'Outside'))
    print("  Probability: {:.4f}".format(prediction))
    print()

print("Total predictions made: {}".format(len(predictions)))
