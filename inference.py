import numpy as np
from pycoral.utils import edgetpu
from pycoral.adapters import common

# Load the model
interpreter = edgetpu.make_interpreter('environmentModel_quantized.tflite')
interpreter.allocate_tensors()

# Prepare input data
input_data = np.array([60, 30504, 101, 26, 48.57692308, 77, 39, -81.71794872, -66], dtype=np.float32)
input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension

# Set input tensor
input_tensor_index = interpreter.get_input_details()[0]['index']
interpreter.set_tensor(input_tensor_index, input_data)

# Run inference
interpreter.invoke()

# Get output tensor
output_tensor_index = interpreter.get_output_details()[0]['index']
output_data = interpreter.get_tensor(output_tensor_index)

# Process the output (depends on your model's output format)
print(f"Output: {output_data}")