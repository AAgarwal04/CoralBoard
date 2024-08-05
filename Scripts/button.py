import tkinter as tk
from tkinter import simpledialog
import threading
import time
import subprocess
import asyncio
from bleak import BleakScanner
import warnings

warnings.filterwarnings("ignore")

location = "Inside"

# result = subprocess.run(["powershell", "-Command", "$env:BLINKA_MCP2221=1"], capture_output=True, text=True)

# Initialize the I2C bus
i2c = busio.I2C(board.SCL, board.SDA)

# HDC2010 I2C address and register addresses
HDC2010_ADDR = 0x40
TEMP_LOW = 0x00
TEMP_HIGH = 0x01
HUMID_LOW = 0x02
HUMID_HIGH = 0x03
CONFIG = 0x0E
MEASUREMENT_CONFIG = 0x0F

def get_wifi_info():
    try:
        output = subprocess.check_output(['netsh', 'wlan', 'show', 'networks', 'mode=Bssid'], encoding='utf-8')
        lines = output.split('\n')
        signal_strengths = []
        for line in lines:
            if 'Signal' in line:
                strength = int(line.split(':')[1].strip().replace('%', ''))
                signal_strengths.append(strength)
        if len(signal_strengths) == 0:
            return 0, 0, 0
        else:
            return len(signal_strengths), max(signal_strengths), sum(signal_strengths)/len(signal_strengths)
    except subprocess.CalledProcessError:
        return 0, 0, 0

async def scan_bluetooth():
    try:
        devices = await BleakScanner.discover(timeout=3.0)
        rssi_strengths = [device.rssi for device in devices if device.rssi is not None]
        if len(rssi_strengths) == 0:
            return 0, 0, 0
        else:
            return len(rssi_strengths), max(rssi_strengths), sum(rssi_strengths)/len(rssi_strengths)
    except Exception as e:
        print(f"Error scanning Bluetooth: {e}")
        return 0, 0, 0

def show_custom_popup(root):
    global location
    popup = tk.Toplevel(root)
    popup.title("Location")
    popup.geometry("200x100")
    
    def on_button_click(choice):
        global location
        location = choice
        popup.destroy()
    
    label = tk.Label(popup, text="Where are you?")
    label.pack(pady=10)
    
    inside_button = tk.Button(popup, text="Inside", command=lambda: on_button_click("Inside"))
    inside_button.pack(side=tk.LEFT, padx=10, pady=10)
    
    outside_button = tk.Button(popup, text="Outside", command=lambda: on_button_click("Outside"))
    outside_button.pack(side=tk.RIGHT, padx=10, pady=10)

def write_register(register, value):
    i2c.writeto(HDC2010_ADDR, bytes([register, value]))

def read_register_2_bytes(register):
    i2c.writeto(HDC2010_ADDR, bytes([register]))
    result = bytearray(2)
    i2c.readfrom_into(HDC2010_ADDR, result)
    return result

# Initialize sensor
write_register(CONFIG, 0x10)  # 14 bit temp, 14 bit humidity

def read_temperature_humidity():
    # Trigger measurement
    write_register(MEASUREMENT_CONFIG, 0x01)
    
    time.sleep(0.05)  # Wait for measurement to complete
    
    # Read Temperature
    temp_data = read_register_2_bytes(TEMP_LOW)
    temp_raw = (temp_data[1] << 8) | temp_data[0]
    temperature = (temp_raw / 65536.0) * 165.0 - 40.0
    
    # Read Humidity
    humid_data = read_register_2_bytes(HUMID_LOW)
    humid_raw = (humid_data[1] << 8) | humid_data[0]
    humidity = (humid_raw / 65536.0) * 100.0
    
    return temperature, humidity

def print_location():
    global location
    filename = "/Users/AgAr082/Documents/Scripts/file.txt"
    
    while True:
        file1 = open(filename, "a")
        time.sleep(1)
        wifiAmnt, wifiMax, wifiAvg = get_wifi_info()
        bleAmnt, bleMax, bleAvg = asyncio.run(scan_bluetooth())
        temp, humid = read_temperature_humidity()
        print(f"Wi-Fi: {wifiAmnt}, {wifiMax}, {wifiAvg:.2f}")
        print(f"Bluetooth: {bleAmnt}, {bleMax}, {bleAvg:.2f}")
        print(f"Temp: {temp}, Humidity: {humid}")
        print(f"You are {location}")
        print("----")
        message = f"{wifiAmnt}, {wifiMax}, {wifiAvg:.2f}, {bleAmnt}, {bleMax}, {bleAvg:.2f}, {temp}, {humid}, {location}\n"
        file1.write(message)
        file1.flush()
        file1.close()
        time.sleep(2)


def main():
    root = tk.Tk()
    root.title("Custom Popup Example")
    root.geometry("300x200")
    
    

    button = tk.Button(root, text="Location", command=lambda: show_custom_popup(root))
    button.pack(pady=20)

    location_thread = threading.Thread(target=print_location, daemon=True)
    location_thread.start()

    root.mainloop()

if __name__ == "__main__":
    main()