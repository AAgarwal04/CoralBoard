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

    while True:
        # Indefinitely update display and upload to cloud.
        sensors = {}
        read_period = int(args.upload_delay / (2 * args.display_duration))
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

            # Load model and prepare interpreter
            interpreter = edgetpu.make_interpreter('enviroModel_edgetpu.tflite')
            interpreter.allocate_tensors()

            # Prepare input data
            arr = []
            input_data = np.array([sensors["humidity"], sensors["ambient_light"], sensors["pressure"], 
                                   sensors["wifiAmnt"], sensors["wifiAvg"], sensors["wifiMax"], sensors['bleAmnt'], sensors["bleMax"], sensors["bleAvg"]], dtype=np.float32)
            msg = " ".join(input_data)
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
                msg = "Outside " + str(num)
            else:
                msg = "Inside " + str(num)
            update_display(enviro.display, msg)
            sleep(30)

            if button.read() == False:
                #file1.close()
                update_display(enviro.display, "Program Terminated :)")
                # print("Program Terminated")
                sleep(2)
                break
        break

if __name__ == '__main__':
    sleep(10)
    thermFile = "/sys/class/thermal/thermal_zone0/trip_point_4_temp"
    file = open(thermFile, "w")
    file.write("25000")
    file.flush()
    file.close()
    main()