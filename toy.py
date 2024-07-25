import numpy as np
from pycoral.utils import edgetpu
from coral.enviro.board import EnviroBoard
from coral.cloudiot.core import CloudIot
from luma.core.render import canvas
from PIL import ImageDraw
from time import sleep
import argparse
import itertools
import threading
import os
import keyboard
from periphery import GPIO, Serial
import shutil
import subprocess
import asyncio
import warnings
from bleak import BleakScanner
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate
import numpy as np
import joblib

DEFAULT_CONFIG_LOCATION = os.path.join(os.path.dirname(__file__), 'cloud_config.ini')

def update_display(display, msg):
    with canvas(display) as draw:
        draw.text((0, 0), msg, fill='white')

def _none_to_nan(val):
    return float('nan') if val is None else val

def get_wifi_info():
    output = subprocess.check_output(['nmcli', '-f', 'SIGNAL', 'dev', 'wifi', 'list'], encoding='utf-8')
    output = output.split('\n')[1:]
    output = list(filter(None, output))
    # output = output[:len(output)-1]
    signal_strengths = []
    for line in output:
        if line[0].isdigit():
            signal_strengths.append(int(line.strip()))
    signal_strengths = list(filter(None, signal_strengths))
    #print(signal_strengths)
    avg = (sum(signal_strengths)/len(signal_strengths)) if len(signal_strengths) != 0 else 0
    if (len(signal_strengths) == 0):
        return 0, 0, 0
    else:
        return len(signal_strengths), avg, max(signal_strengths)

async def scan_bluetooth():
    scanner = BleakScanner()
    devices = await scanner.discover(timeout=3.0)
    rssiStrength = []
    for device in devices:
        rssi = device.rssi if hasattr(device, 'rssi') else "Unknown"
        rssiStrength.append(rssi)
    rssiStrength = list(filter(lambda x: x != "Unknown", rssiStrength))
    avg = sum(rssiStrength)/len(rssiStrength) if len(rssiStrength) != 0 else 0
    if (len(rssiStrength) == 0):
        return 0, 0, 0
    else:
        return len(rssiStrength), max(rssiStrength), avg

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
    
    input_scale, input_zero_point = input_details[0]['quantization']
    row_quantized = X_scaled[0] / input_scale + input_zero_point
    row_quantized = row_quantized.astype(input_details[0]['dtype'])
    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(row_quantized, axis=0))
    
    interpreter.invoke()
    
    output = interpreter.get_tensor(output_details[0]['index'])
    
    output_scale, output_zero_point = output_details[0]['quantization']
    output = (output.astype(np.float32) - output_zero_point) * output_scale
    
    return output[0]

def main():
    # Pull arguments from command line.
    parser = argparse.ArgumentParser(description='Enviro Kit Demo')
    parser.add_argument('--display_duration',
            help='Measurement display duration (seconds)', type=int,
            default=5)
    parser.add_argument('--upload_delay', help='Cloud upload delay (seconds)',
            type=int, default=300)
    parser.add_argument(
            '--cloud_config', help='Cloud IoT config file', default=DEFAULT_CONFIG_LOCATION)
    args = parser.parse_args()
    # Create instances of EnviroKit and Cloud IoT.
    enviro = EnviroBoard()
    # button = GPIO("/dev/gpiochip2", 9, "in")

    # Load the model and scaler
    model_path = 'environmentModel_quantized_edgetpu.tflite'
    scaler_path = 'scaler.pkl'
    interpreter = load_model(model_path)
    scaler = load_scaler(scaler_path)

    while True:
        # Indefinitely update display and upload to cloud.
        sensors = {}
        # read_period = int(args.upload_delay / (2 * args.display_duration))
        # flag = True if (input("Inside or Outside (1 or 0): ") == '1') else False
        num = 0

        while True:
            sensors['temperature'] = int(enviro.temperature)
            sensors['humidity'] = int(enviro.humidity)
            sensors['ambient_light'] = int(enviro.ambient_light)
            sensors['pressure'] = int(enviro.pressure)
            sensors['wifiAmnt'], sensors['wifiAvg'], sensors['wifiMax'] = get_wifi_info()
            sensors['bleAmnt'], sensors['bleMax'], sensors['bleAvg'] = asyncio.run(scan_bluetooth())
            warnings.filterwarnings("ignore", category=FutureWarning)
            input_data = np.array([sensors["humidity"], sensors["ambient_light"], sensors["pressure"], 
                                   sensors["wifiAmnt"], sensors["wifiAvg"], sensors["wifiMax"], sensors['bleAmnt'], sensors["bleMax"], sensors["bleAvg"]], dtype=np.float32)
            msg = ", ".join(input_data)
            print(msg)

            # Perform inference
            prediction = predict(interpreter, scaler, X)

            features = ["RH", "Light", "WifiAmnt", "WifiAvg", "WifiMax", "BLEAmnt", "BLEAvg", "BLEMax"]

            # Print results
            print(f"Row 1:")
            print(f"  Raw Data:")
            for feature, value in zip(features, X[0]):
                print(f"    {feature}: {value}")
            print(f"  Prediction: {'Inside' if prediction > 0.5 else 'Outside'}")
            print(f"  Probability: {prediction:.4f}")
            print()

            print(f"Total predictions made: 1")
            num += 1
            sleep(5)

if __name__ == "__main__":
    main()