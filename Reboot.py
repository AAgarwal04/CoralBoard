#!/usr/bin/env python3
from time import sleep
import os
import shutil
import subprocess

def reboot_board():
    subprocess.run(["sudo", "reboot"])

sleep(65*60)
reboot_board()