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
import tflite_runtime.interpreter as tflite

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

def scale_input(input_data, mean, std):
    return (input_data - mean) / std

def load_model(model_path):
    interpreter = tflite.Interpreter(model_path)
    interpreter.allocate_tensors()
    return interpreter

def predict(interpreter, input_data):
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    mean = np.array([62.02942646, 5399.85434153, 152.20011491, 43.37125782, 81.11298461, 42.80942746, -81.13534295, -60.04821143], dtype=np.float32)
    std = np.array([1.10869471e+01, 1.71231261e+04, 6.20636820e+01, 8.15061296e+00, 1.65243413e+01, 1.62102902e+01, 2.94259457e+00, 9.82795009e+00], dtype=np.float32)

    input_data_scaled = scale_input(input_data, mean, std)
    input_data_scaled = input_data_scaled.reshape(input_details['shape'])

    interpreter.set_tensor(input_details['index'], input_data_scaled)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details['index'])
    return "Inside" if output_data[0][0] > 0.5 else "Outside"

def main():
    model_path = 'environmentModel.tflite'
    interpreter = load_model(model_path)

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
    while True:
        # Indefinitely update display and upload to cloud.
        sensors = {}
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

            # Prepare input data
            arr = []
            input_data = np.array([sensors["humidity"], sensors["ambient_light"], 
                                   sensors["wifiAmnt"], sensors["wifiAvg"], sensors["wifiMax"], sensors['bleAmnt'], sensors["bleMax"], sensors["bleAvg"]], dtype=np.float32)
            
            prediction = predict(interpreter, input_data)
            msg = " ".join(input_data)
            print(msg)
            print(f"Prediction: {prediction}")
            num += 1
            sleep(5)

    
    # # Example input data
    # input_data = np.array([43, 1354, 139, 46.338, 82, 78, -79.967, -38], dtype=np.float32)
    
    # prediction = predict(interpreter, input_data)
    # print(f"Prediction: {prediction}")

if name == "main":
    main()
