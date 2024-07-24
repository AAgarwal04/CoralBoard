import numpy as np
from pycoral.utils import edgetpu

# Load model and prepare interpreter
interpreter = edgetpu.make_interpreter('environmentModel_edgetpu.tflite')
interpreter.allocate_tensors()

# Prepare input data
input_data = np.array([43, 1354, 139, 46.338, 82, 78, -79.967, -38], dtype=np.float32)
input_details = interpreter.get_input_details()[0]
input_data = input_data.reshape(input_details['shape'])

# Ensure input data is of type FLOAT32
input_data = input_data.astype(np.float32)

# Run inference
interpreter.set_tensor(input_details['index'], input_data)
interpreter.invoke()

# Get output
output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
class_id = np.argmax(output_data)
score = output_data[0][class_id]

print(f"Class: {class_id}, Score: {score}")
