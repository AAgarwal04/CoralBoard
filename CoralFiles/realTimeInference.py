import numpy as np
from pycoral.utils import edgetpu
from coral.enviro.board import EnviroBoard
from coral.cloudiot.core import CloudIot
from luma.core.render import canvas
from PIL import ImageDraw
from time import sleep
import argparse
import threading
import os
import keyboard
from periphery import GPIO, Serial
import subprocess
import asyncio
import warnings
from bleak import BleakScanner
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate
import joblib

DEFAULT_CONFIG_LOCATION = os.path.join(os.path.dirname(__file__), 'cloud_config.ini')

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def update_display(display, msg):
    with canvas(display) as draw:
        draw.text((0, 0), msg, fill='white')

def _none_to_nan(val):
    return float('nan') if val is None else val

def get_wifi_info():
    output = subprocess.check_output(['nmcli', '-f', 'SIGNAL', 'dev', 'wifi', 'list'], encoding='utf-8')
    output = output.split('\n')[1:]
    output = list(filter(None, output))
    signal_strengths = []
    for line in output:
        if line[0].isdigit():
            signal_strengths.append(int(line.strip()))
    signal_strengths = list(filter(None, signal_strengths))
    avg = (sum(signal_strengths)/len(signal_strengths)) if len(signal_strengths) != 0 else 0
    if len(signal_strengths) == 0:
        return 0, 0, 0
    else:
        return len(signal_strengths), avg, max(signal_strengths)

async def scan_bluetooth():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        scanner = BleakScanner()
        devices = await scanner.discover(timeout=3.0)
    rssiStrength = []
    for device in devices:
        rssi = device.rssi if hasattr(device, 'rssi') else "Unknown"
        rssiStrength.append(rssi)
    rssiStrength = list(filter(lambda x: x != "Unknown", rssiStrength))
    avg = sum(rssiStrength)/len(rssiStrength) if len(rssiStrength) != 0 else 0
    if len(rssiStrength) == 0:
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
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="bleak")
    
    parser = argparse.ArgumentParser(description='Enviro Kit Demo')
    parser.add_argument('--display_duration', help='Measurement display duration (seconds)', type=int, default=5)
    parser.add_argument('--upload_delay', help='Cloud upload delay (seconds)', type=int, default=300)
    parser.add_argument('--cloud_config', help='Cloud IoT config file', default=DEFAULT_CONFIG_LOCATION)
    args = parser.parse_args()
    
    enviro = EnviroBoard()

    model_path = '../environmentModel_quantized_edgetpu.tflite'
    scaler_path = '../scaler.pkl'
    interpreter = load_model(model_path)
    scaler = load_scaler(scaler_path)

    while True:
        sensors = {}
        num = 0
        while True:
            try:
                sensors['humidity'] = int(enviro.humidity)
                sensors['ambient_light'] = int(enviro.ambient_light)
                sensors['wifiAmnt'], sensors['wifiAvg'], sensors['wifiMax'] = get_wifi_info()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    sensors['bleAmnt'], sensors['bleMax'], sensors['bleAvg'] = asyncio.run(scan_bluetooth())

                input_data = np.array([
                    sensors["humidity"], sensors["ambient_light"],
                    sensors["wifiAmnt"], sensors["wifiAvg"], sensors["wifiMax"],
                    sensors['bleAmnt'], sensors["bleAvg"], sensors["bleMax"]
                ], dtype=np.float32)

                prediction = predict(interpreter, scaler, np.array([input_data]))
                val = "Inside" if prediction > 0.5 else "Outside"
                msg = str(num) + ": " + val
                # print(max(scan_results), sum(scan_results)/len(scan_results))
                update_display(enviro.display, msg)
                # print("Inside" if prediction > 0.5 else "Outside")

                sleep(2)
            except Exception as e:
                msg = "Error"
                # print(f"An error occurred: {e}")
                update_display(enviro.display, msg)
                sleep(2)

if __name__ == "__main__":
    main()