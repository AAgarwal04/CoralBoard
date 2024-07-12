import numpy as np
from pycoral.utils import edgetpu
from pycoral.adapters import common
from pycoral.adapters import classify

# Load the model
interpreter = edgetpu.make_interpreter('enviroModel_edgetpu.tflite')
interpreter.allocate_tensors()

# Your input array
input_data = np.array([56, 11192, 100, 0.27, 200, 44.805, 97, 115, -77.70434782608696], dtype=np.float32)

# Reshape and normalize the input if necessary
input_data = input_data.reshape(1, -1)  # Reshape to (1, 9) for batch size 1

input_details = interpreter.get_input_details()[0]
if input_details['dtype'] == np.uint8:
    input_scale, input_zero_point = input_details['quantization']
    input_data = input_data / input_scale + input_zero_point
    input_data = input_data.astype(np.uint8)

common.set_input(interpreter, input_data)

interpreter.invoke()
output = classify.get_classes(interpreter, top_k=1)[0]

if output.id == 0:
    print("Outside")
else:
    print("Inside")