import subprocess
import time
import asyncio
from bleak import BleakScanner
import warnings

warnings.filterwarnings("ignore")

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

def main():
    while True:
        # Wi-Fi scanning
        wifiAmnt, wifiMax, wifiAvg = get_wifi_info()
        print(f"Wi-Fi: {wifiAmnt}, {wifiMax}, {wifiAvg:.2f}")

        # Bluetooth scanning
        bleAmnt, bleMax, bleAvg = asyncio.run(scan_bluetooth())
        print(f"Bluetooth: {bleAmnt}, {bleMax}, {bleAvg:.2f}")

        print("---")
        time.sleep(5)  # Wait for 5 seconds before the next scan

if __name__ == "__main__":
    main()