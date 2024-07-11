import numpy as np
from pycoral.utils import edgetpu

# Load the model
interpreter = edgetpu.make_interpreter('environmentModel_quantized.tflite')
interpreter.allocate_tensors()

# Get input details
input_details = interpreter.get_input_details()[0]
input_shape = input_details['shape']
input_scale, input_zero_point = input_details['quantization']

print(f"Expected input shape: {input_shape}")

# Prepare input data
input_data = np.array([[60, 30504, 101, 26, 48.57692308, 77, 39, -81.71794872, -66]], dtype=np.float32)

# Reshape input data to match expected shape
input_data = input_data.reshape(input_shape)

# Quantize the input data to int8
input_data = input_data / input_scale + input_zero_point
input_data = np.clip(np.round(input_data), -128, 127).astype(np.int8)

print(f"Adjusted input shape: {input_data.shape}")
print(f"Adjusted input dtype: {input_data.dtype}")

# Set input tensor
interpreter.set_tensor(input_details['index'], input_data)

# Run inference
interpreter.invoke()

# Get output
output_details = interpreter.get_output_details()[0]
output_data = interpreter.get_tensor(output_details['index'])

# Dequantize the output if necessary
if output_details['dtype'] == np.int8:
    output_scale, output_zero_point = output_details['quantization']
    output_data = output_scale * (output_data.astype(np.float32) - output_zero_point)

# Print the result
print(f"Output: {output_data}")
