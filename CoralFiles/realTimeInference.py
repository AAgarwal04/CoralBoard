import numpy as np
from pycoral.utils import edgetpu
from pycoral.adapters import common
from pycoral.adapters import classify
from coral.enviro.board import EnviroBoard
from coral.cloudiot.core import CloudIot
from luma.core.render import canvas
from PIL import ImageDraw
from time import sleep
import argparse
import os
from periphery import GPIO, Serial
import subprocess
import asyncio
import warnings
from bleak import BleakScanner
import warnings

warnings.filterwarnings("ignore", message="BLEDevice.rssi is deprecated", category=FutureWarning)

DEFAULT_CONFIG_LOCATION = os.path.join(os.path.dirname(__file__), 'cloud_config.ini')

def update_display(display, msg):
    with canvas(display) as draw:
        draw.text((0, 0), msg, fill='white')

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
    return len(rssiStrength), max(rssiStrength), avg

# Load the saved EdgeTPU model
model_path = 'environmentModel_edgetpu.tflite'
interpreter = edgetpu.make_interpreter(model_path)
interpreter.allocate_tensors()

# Get input details
input_details = interpreter.get_input_details()
input_shape = input_details[0]['shape']
# print("Expected input shape:", input_shape)

def predict(X):
    # Ensure X is a 2D array
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    # Convert to float32
    input_data = X.astype(np.float32)
    
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    interpreter.invoke()
    
    # Get the output
    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data.flatten()[0]  # Flatten and take the first element


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
    button = GPIO("/dev/gpiochip2", 9, "in")
    num = 0

    while True:
        wifiAmnt, wifiAvg, wifiMax = get_wifi_info()
        bleAmnt, bleMax, bleAvg = asyncio.run(scan_bluetooth())
        X = np.array([
            int(enviro.humidity), int(enviro.ambient_light), int(enviro.pressure), 
            wifiAmnt, wifiAvg, wifiMax, 
            bleAmnt, bleAvg, bleMax
        ])

        # Perform inference
        prediction = predict(X)
        features = ["RH", "Light", "Pressure", "WifiAmnt", "WifiAvg", "WifiMax", "BLEAmnt", "BLEAvg", "BLEMax"]
        outcome = 'Inside' if prediction > 0.5 else 'Outside'

        # # Print results
        # print("Row {}:".format(num))
        # print("  Raw Data:")
        # for feature, value in zip(features, X):
        #     print("    {}: {}".format(feature, value))
        # print("  Prediction: {}".format(outcome))
        # print("  Probability: {:.4f}".format(prediction))

        update_display(enviro.display, str(num) + ": " + outcome)

        sleep(3)
        num += 1

if __name__ == '__main__':
    thermFile = "/sys/class/thermal/thermal_zone0/trip_point_4_temp"
    file = open(thermFile, "w")
    file.write("20000")
    file.flush()
    file.close()
    main()