#!/usr/bin/env python3

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

def reboot_board():
    subprocess.run(["sudo", "reboot"])

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
        flag = True
        #file1 = open("log.txt", "a")
        num = 0
        filename = "/home/mendel/logFiles/file.txt"
        file1 = open(filename, "a")
        message = ""

        while True:
            # file1 = open("log.txt", "a")   
            # file1.write(str(num) + " ")
            num += 1
            # First display temperature and RH.
            sensors['temperature'] = enviro.temperature
            sensors['humidity'] = enviro.humidity
            #msg = 'Temp: %.2f C\n' % _none_to_nan(sensors['temperature'])
            #msg += 'RH: %.2f %%' % _none_to_nan(sensors['humidity'])
            msg = "T: " + str(int(enviro.temperature)) + " R: " + str(int(enviro.humidity)) + " L: " + str(int(enviro.ambient_light)) + '\n'
            message += str(num) + " Temp: " + str(int(enviro.temperature)) + " RH: " + str(int(enviro.humidity)) + " "
            #file1.write("Temp: " + str(int(enviro.temperature)) + " RH: " + str(int(enviro.humidity)) + " ")
            sensors['ambient_light'] = enviro.ambient_light
            sensors['pressure'] = enviro.pressure
            wifiAmnt, wifiAvg, wifiMax = get_wifi_info()
            #msg = 'Light: %.2f lux\n' % _none_to_nan(sensors['ambient_light'])
            #msg += 'Pressure: %.2f kPa' % _none_to_nan(sensors['pressure'])
            uv = ((enviro.grove_analog * 5 /409.6) * 1000 / 4.3) / 21

            warnings.filterwarnings("ignore", category=FutureWarning)
            bleAmnt, bleMax, bleAvg = asyncio.run(scan_bluetooth())

            msg += " P: " + str(int(enviro.pressure)) + " U: " + str(round(uv,2))
            message += "Light: " + str(int(enviro.ambient_light)) + " Pressure: " + str(int(enviro.pressure)) + " UV: " + str(round(uv, 2))
            message += " Amnt: " + str(wifiAmnt) + " Avg: " + str(wifiAvg) + " Max: " + str(wifiMax)
            message += " BleAmnt: " + str(bleAmnt) + " BleAvg: " + str(bleAvg) + " BleMax: " + str(bleMax)
            #file1.write("Light: " + str(int(enviro.ambient_light)) + " Pressure: " + str(int(enviro.pressure)) + " UV: " + str(round(uv, 2)))
            #flag = ~flag if (~button.read()) else flag
            #file1.write(" Inside\n" if (flag) else " Outside\n")
            # update_display(enviro.display, msg)
            #if button.read() == False:
            #    if flag == True: flag = False
            #    else: flag = True
            #    #uart1.write(b"Tyson")
            val = "Inside" if flag else "Outside"
            msg += " " + val
            #print(val)
            path = "/usr/lib/python3/dist-packages/coral/enviro"
            #print(shutil.disk_usage(path))
            message += " " + val + "\n"
            # file1.write(" " + val + "\n")
            # update_display(enviro.display, msg)
            msg = str(num)
            update_display(enviro.display, msg)
            sleep(26)
            if (num % 30 == 0):
                file1.write(message)
                file1.flush()
                message = ""
                if num % 60 == 0:
                    file1.close()
                    reboot_board()
            sleep(27)

            if button.read() == False:
                #file1.close()
                update_display(enviro.display, "Program Terminated :)")
                # print("Program Terminated")
                sleep(2)
                break
        break

if __name__ == '__main__':
    sleep(5)
    thermFile = "/sys/class/thermal/thermal_zone0/trip_point_4_temp"
    file = open(thermFile, "w")
    file.write("25000")
    file.flush()
    file.close()
    main()
