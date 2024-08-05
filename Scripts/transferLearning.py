import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import time
import os

def check_and_process_file(file_path, min_lines=1000):
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist. Waiting...")
        return None, None

    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    if len(lines) < min_lines:
        print(f"Not enough data (current: {len(lines)}, required: {min_lines}). Waiting...")
        return None, None
    
    data = []
    labels = []
    for line in lines:
        values = line.strip().split(',')
        features = [float(val.strip()) for val in values[:6]]
        label = 1 if values[-1].strip() == 'Inside' else 0
        data.append(features)
        labels.append(label)
    
    # Clear the file contents
    open(file_path, 'w').close()
    
    print(f"Processed {len(lines)} lines of data.")
    return np.array(data), np.array(labels)

def transfer_learning(X_new, y_new):
    # Load the existing model
    model = keras.models.load_model('environmentModel.h5')

    # Load the scaler
    scaler = joblib.load('scaler.pkl')

    # Scale the new data using the loaded scaler
    X_new_scaled = scaler.transform(X_new)

    # Split the new data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_new_scaled, y_new, test_size=0.2, random_state=42)

    # Prepare the data for training
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)

    # Fine-tune the model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_dataset,
                        epochs=50,
                        validation_data=val_dataset,
                        callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])

    # Save the fine-tuned model
    model.save('environmentModel.h5')

    print("Fine-tuned model saved successfully.")

    # Evaluate the fine-tuned model
    test_loss, test_accuracy = model.evaluate(val_dataset)
    print(f"Test accuracy: {test_accuracy:.4f}")

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

def main():
    file_path = 'file.txt'
    while True:
        X_new, y_new = check_and_process_file(file_path)
        if X_new is not None and y_new is not None:
            transfer_learning(X_new, y_new)
            convert()
            
        else:
            print("Waiting for 10 minute before checking again...")
            time.sleep(600)  # Wait for 5 minutes

if __name__ == "__main__":
    main()
