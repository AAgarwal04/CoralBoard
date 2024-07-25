#!/usr/bin/env python3
from time import sleep
import os
import shutil
import subprocess

def reboot_board():
    subprocess.run(["sudo", "reboot"])


sleep(5)
thermFile = "/sys/class/thermal/thermal_zone0/trip_point_4_temp"
file = open(thermFile, "w")
file.write("25000")
file.flush()
file.close()
sleep(25*60)
reboot_board()
