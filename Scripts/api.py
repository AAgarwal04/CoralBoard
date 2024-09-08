import requests
import base64
import subprocess

class Wifi:
    def __init__(self, ssid=None, bssid=None, signalStrength=None):
        self.ssid = ssid
        self.bssid = bssid
        self.signalStrength = signalStrength
    
    def setSsid(self, ssid):
        self.ssid = ssid
    
    def setBssid(self, bssid):
        self.bssid = bssid

    def setSignalStrength(self, signalStrength):
        self.signalStrength = signalStrength
    
    def display(self):
        return f"SSID: {self.ssid}, BSSID: {self.bssid}, Signal Strength: {self.signalStrength}"

# def search_wigle(ssid, bssid):
#     url = "https://api.wigle.net/api/v2/network/search"
    
#     params = {
#         "ssid": ssid,
#         "netid": bssid
#     }
    
#     auth_string = base64.b64encode(f"AIDd12d8bace0d1ba84c562ac896e184535:cb0bc3b70de92386714507071312b6ec".encode()).decode()
#     headers = {
#         "Authorization": f"Basic {auth_string}"
#     }
    
#     response = requests.get(url, params=params, headers=headers)
    
#     if response.status_code == 200:
#         data = response.json()
#         if data.get("success") and data.get("results"):
#             result = data["results"][0]
#             return result.get("trilat"), result.get("trilong")
    
#     return None, None

def search_wigle(ssid, bssid=None):
    url = "https://api.wigle.net/api/v2/network/search"
    
    params = {
        "ssid": ssid
    }
    if bssid:
        params["netid"] = bssid
    auth_string = base64.b64encode(f"AIDd12d8bace0d1ba84c562ac896e184535:cb0bc3b70de92386714507071312b6ec".encode()).decode()
    headers = {
        "Authorization": f"Basic {auth_string}"
    }
    
    response = requests.get(url, params=params, headers=headers)
    
    print(f"Searching for SSID: {ssid}" + (f", BSSID: {bssid}" if bssid else ""))
    print(f"Response status code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"API Response: {data}")
        if data.get("success") and data.get("results"):
            result = data["results"][0]
            lat = result.get("trilat")
            lon = result.get("trilong")
            print(f"Found location: Lat {lat}, Lon {lon}")
            return lat, lon
        else:
            print("No results found in the API response")
    else:
        print(f"API request failed with status code: {response.status_code}")
    
    return None, None

def get_wifi_info():
    try:
        output = subprocess.check_output(['netsh', 'wlan', 'show', 'networks', 'mode=Bssid'])
        try:
            output = output.decode('utf-8')
        except UnicodeDecodeError:
            output = output.decode('latin-1')
        
        lines = output.split('\n')
        
        networks = []
        currentWifi = None

        for line in lines:
            line = line.strip()
            if 'SSID' in line and ':' in line:
                if currentWifi:
                    networks.append(currentWifi)
                currentWifi = Wifi()
                currentWifi.setSsid(line.split(':')[1].strip())
            elif 'BSSID' in line and ':' in line:
                currentWifi.setBssid(line.split(':')[1].strip())
            elif 'Signal' in line:
                currentWifi.setSignalStrength(line.split(':')[1].strip())
        
        if currentWifi:
            networks.append(currentWifi)

        if len(networks) == 0:
            print("No networks found")
            return []
        else:
            print(f"Found {len(networks)} networks:")
            for i, network in enumerate(networks):
                print(f"Network {i+1}: {network.display()}")
            
            # Calculate statistics
            signal_strengths = [int(network.signalStrength.replace('%', '')) for network in networks if network.signalStrength]
            if signal_strengths:
                return len(signal_strengths), max(signal_strengths), sum(signal_strengths)/len(signal_strengths)
            else:
                return 0, 0, 0

    except subprocess.CalledProcessError:
        print("Error executing netsh command")
        return []

# Usage
amnt, maxVal, avgVal = get_wifi_info()
    
# if amnt > 0:
#     for x in range(len(wifi['network']) - 1):
#         print(f"\nProcessing network {x+1} of {len(wifi['SSID'])}")
#         latitude, longitude = search_wigle(wifi['SSID'][x], wifi['BSSID'][x])
#         if latitude and longitude:
#             print(f"Location found: Latitude {latitude}, Longitude {longitude}")
#         else:
#             print("Location not found")
#         print("---")
# else:
#     print("No WiFi networks found")