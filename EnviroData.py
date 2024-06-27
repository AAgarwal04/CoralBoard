# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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

DEFAULT_CONFIG_LOCATION = os.path.join(os.path.dirname(__file__), 'cloud_config.ini')

def update_display(display, msg):
    with canvas(display) as draw:
        draw.text((0, 0), msg, fill='white')

def _none_to_nan(val):
    return float('nan') if val is None else val

def watchdog():
    os.system('reboot')

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
    uart1 = Serial("/dev/ttymxc0", 115200)
    with CloudIot(args.cloud_config) as cloud:
        # Indefinitely update display and upload to cloud.
        sensors = {}
        read_period = int(args.upload_delay / (2 * args.display_duration))
        flag = True if (input("Inside or Outside (1 or 0): ") == '1') else False
        #file1 = open("log.txt", "a")
        num = 0
        filenum = 0
        filename = "file" + str(filenum) + ".txt"
        file1 = open(filename, "a")
        message = ""

        watchdog_timer = threading.Timer(10 * 60, watchdog)
        watchdog_timer.start()

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
            message += "Temp: " + str(int(enviro.temperature)) + " RH: " + str(int(enviro.humidity)) + " "
            #file1.write("Temp: " + str(int(enviro.temperature)) + " RH: " + str(int(enviro.humidity)) + " ")
            sensors['ambient_light'] = enviro.ambient_light
            sensors['pressure'] = enviro.pressure
            #msg = 'Light: %.2f lux\n' % _none_to_nan(sensors['ambient_light'])
            #msg += 'Pressure: %.2f kPa' % _none_to_nan(sensors['pressure'])
            uv = ((enviro.grove_analog * 5 /409.6) * 1000 / 4.3) / 21
            msg += " P: " + str(int(enviro.pressure)) + " U: " + str(round(uv,2))
            message += "Light: " + str(int(enviro.ambient_light)) + " Pressure: " + str(int(enviro.pressure)) + " UV: " + str(round(uv, 2))
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
            sleep(30)
            if (num % 20 == 0):
                file1.write(message)
                file1.flush()
                message = ""
                if num % 100 == 0:
                    file1.close()
                    filenum += 1
                    filename = "file" + str(filenum) + ".txt"
                    file1 = open(filename, "a")

            sleep(27)

            if button.read() == False:
                #file1.close()
                update_display(enviro.display, "Program Terminated :)")
                print("Program Terminated")
                sleep(2)
                break
            
            watchdog_timer.cancel()
            watchdog_timer = threading.Timer(10 * 60, watchdog)
            watchdog_timer.start()

if __name__ == '__main__':
    main()
