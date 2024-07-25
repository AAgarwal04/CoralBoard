from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate
import numpy as np
import joblib

def load_model(model_path):
    edgetpu_delegate = load_delegate('libedgetpu.so.1')
    interpreter = Interpreter(
        model_path=model_path,
        experimental_delegates=[edgetpu_delegate])
    interpreter.allocate_tensors()
    return interpreter

def load_scaler(scaler_path):
    return joblib.load(scaler_path)

def preprocess_data(scaler, X):
    return scaler.transform(X)

def predict(interpreter, scaler, X):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    X_scaled = preprocess_data(scaler, X)
    results = []
    for row in X_scaled:
        input_scale, input_zero_point = input_details[0]['quantization']
        row_quantized = row / input_scale + input_zero_point
        row_quantized = row_quantized.astype(input_details[0]['dtype'])
        interpreter.set_tensor(input_details[0]['index'], np.expand_dims(row_quantized, axis=0))
        
        interpreter.invoke()
        
        output = interpreter.get_tensor(output_details[0]['index'])
        
        output_scale, output_zero_point = output_details[0]['quantization']
        output = (output.astype(np.float32) - output_zero_point) * output_scale
        
        results.append(output[0])
    return np.array(results)

def main():
    # Load the model and scaler
    model_path = 'environmentModel_quantized_edgetpu.tflite'
    scaler_path = 'scaler.pkl'
    interpreter = load_model(model_path)
    scaler = load_scaler(scaler_path)

    # Load and preprocess test data
    X = np.array([
        [53, 499, 171, 48, 87, 65, -81, -57],
        [65, 31000, 28, 50, 80, 40, -80, -65],
        [55, 30000, 24, 47, 75, 38, -83, -67]
    ])

    # Perform inference
    start = 0
    end = min(start + 50, len(X))
    X_subset = X[start:end]

    predictions = predict(interpreter, scaler, X_subset)

    features = ["RH", "Light", "WifiAmnt", "WifiAvg", "WifiMax", "BLEAmnt", "BLEAvg", "BLEMax"]

    # Print results
    for i, (row, prediction) in enumerate(zip(X_subset, predictions)):
        print(f"Row {i+start+1}:")
        print(f"  Raw Data:")
        for feature, value in zip(features, row):
            print(f"    {feature}: {value}")
        print(f"  Prediction: {'Inside' if prediction > 0.5 else 'Outside'}")
        print(f"  Probability: {prediction[0]:.4f}")
        print()

    print(f"Total predictions made: {len(predictions)}")

if __name__ == "__main__":
    main()