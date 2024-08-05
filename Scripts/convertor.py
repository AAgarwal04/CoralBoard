import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

def load_new_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            values = line.strip().split(',')
            features = [float(val.strip()) for val in values[:6]]
            data.append(features)
    return np.array(data)

def get_random_batch(data, batch_size=1):
    indices = np.random.randint(0, data.shape[0], size=batch_size)
    return data[indices]

def representative_data_gen(normalized_data):
    for _ in range(100):  # Generate 100 sample inputs
        sample = get_random_batch(normalized_data)
        yield [sample]

def create_normal_tflite_model(model, output_path):
    normal_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = normal_converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Normal TFLite model saved successfully to {output_path}")

def create_quantized_tflite_model(model, normalized_data, output_path):
    quant_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    quant_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quant_converter.representative_dataset = lambda: representative_data_gen(normalized_data)
    quant_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    quant_converter.inference_input_type = tf.int8
    quant_converter.inference_output_type = tf.int8
    
    tflite_quant_model = quant_converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_quant_model)
    
    print(f"Quantized TFLite model saved successfully to {output_path}")

def convert():
    # Load the existing model
    model = keras.models.load_model('environmentModel.h5')

    # Load data from text file
    text_file_path = 'file.txt'
    data = load_new_data(text_file_path)

    # Normalize the data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data).astype(np.float32)  # Convert to float32

    # Create and save normal TFLite model
    create_normal_tflite_model(model, 'environmentModel.tflite')

    # Create and save quantized TFLite model
    create_quantized_tflite_model(model, normalized_data, 'environmentModel_quantized.tflite')

if __name__ == "__main__":
    convert()