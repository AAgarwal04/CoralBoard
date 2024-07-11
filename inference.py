import numpy as np
from pycoral.utils import edgetpu
from pycoral.adapters import common
from pycoral.adapters import classify

# Load the model
interpreter = edgetpu.make_interpreter('environmentModel_quantized.tflite')
interpreter.allocate_tensors()

# Prepare input data
# Prepare input data
input_data = np.array([[60, 30504, 101, 26, 48.57692308, 77, 39, -81.71794872, -66]], dtype=np.float32)

# Reshape input data to 1D array
input_data = input_data.reshape(-1)

# Quantize the input data to int8
input_details = interpreter.get_input_details()[0]
if input_details['dtype'] == np.int8:
    input_scale, input_zero_point = input_details['quantization']
    input_data = input_data / input_scale + input_zero_point
    input_data = input_data.astype(np.int8)

# Set input tensor
common.set_input(interpreter, input_data)

# Run inference
interpreter.invoke()

# Get output
output_details = interpreter.get_output_details()[0]
output_data = common.output_tensor(interpreter, 0)

# Dequantize the output if necessary
if output_details['dtype'] == np.int8:
    output_scale, output_zero_point = output_details['quantization']
    output_data = output_scale * (output_data - output_zero_point)

# Print the result
print(f"Output: {output_data}")

# If your model is a classifier, you can use the classify adapter
# classes = classify.get_classes(interpreter, top_k=1)
# for c in classes:
#     print(f'Class: {c.id}, Score: {c.score}')