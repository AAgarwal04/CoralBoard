import numpy as np
from pycoral.utils import edgetpu

# Load model and prepare interpreter
interpreter = edgetpu.make_interpreter('enviroModel_edgetpu.tflite')
interpreter.allocate_tensors()

# Prepare input data
input_data = np.array([56, 11192, 100, 0.27, 200, 44.805, 97, 115, -77.70434782608696], dtype=np.float32)
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

if class_id == 0:
    print("Outside")
else:
    print("Inside")
