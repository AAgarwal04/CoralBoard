import time
import board
import busio
import subprocess

result = subprocess.run(["powershell", "-Command", "$env:BLINKA_MCP2221=1"], capture_output=True, text=True)

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

# Main loop
while True:
    temp, humid = read_temperature_humidity()
    print(f"Temperature: {temp:.2f}°C, Humidity: {humid:.2f}%")
    time.sleep(1)
