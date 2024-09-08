import subprocess
import time

def show_and_close_available_networks():
    try:
        # Open the "Show Available Networks" panel
        process = subprocess.Popen(["explorer.exe", "ms-availablenetworks:"])
        print("Show Available Networks panel opened successfully.")
        
        # Wait for 1 second
        time.sleep(1)
        
        # Close the panel
        subprocess.run(["taskkill", "/F", "/IM", "explorer.exe"], check=True)
        print("Show Available Networks panel closed successfully.")
        
        # Restart explorer.exe to restore the desktop and taskbar
        subprocess.Popen(["explorer.exe"])
        print("Explorer restarted.")
        
    except subprocess.CalledProcessError:
        print("Failed to open or close the Show Available Networks panel.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Call the function to show available networks, wait, and close
show_and_close_available_networks()