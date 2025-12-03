import socket
import json
import time
import math
import usb.core
import usb.util
from tuning import Tuning

# --- Configuration ---
ODAS_HOST = '127.0.0.1'  # Localhost (same computer)
ODAS_PORT = 9000         # Port for "Tracked Sources" (SST)

# --- USB Initialization ---
dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
if not dev:
    print("ReSpeaker not found!")
    exit()
mic_tuning = Tuning(dev)

# --- Connect to ODAS Studio ---
print(f"Connecting to ODAS Studio at {ODAS_HOST}:{ODAS_PORT}...")
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    sock.connect((ODAS_HOST, ODAS_PORT))
    print("Connected! Check the ODAS Studio window.")
except ConnectionRefusedError:
    print("Error: Could not connect. Make sure ODAS Studio is OPEN first.")
    exit()

# --- Main Loop ---
print("Streaming tracking data...")

try:
    while True:
        # 1. Get Data from ReSpeaker
        angle_deg = mic_tuning.direction
        is_voice = mic_tuning.is_voice()
        
        # 2. Convert Angle to Unit Vector (X, Y, Z)
        # ODAS expects a vector pointing to the sound on a unit sphere.
        # We map the 2D ReSpeaker angle to the X/Y plane.
        
        angle_rad = math.radians(angle_deg)
        
        # Calculate vector components
        x = math.cos(angle_rad)
        y = math.sin(angle_rad)
        z = 0.0  # ReSpeaker is 2D, so Z is 0 (equator)

        # 3. Create JSON Payload
        # Structure based on ODAS SST (Sound Source Tracking) format
        activity_level = 1.0 if is_voice else 0.0
        
        data = {
            "timeStamp": int(time.time() * 1000),
            "src": [
                {
                    "id": 1,             # Arbitrary ID for the sound source
                    "tag": "voice",
                    "x": x,
                    "y": y,
                    "z": z,
                    "activity": activity_level
                }
            ]
        }

        # 4. Send JSON
        # ODAS expects newline-delimited JSON
        message = json.dumps(data) + "\n"
        sock.sendall(message.encode('utf-8'))
        
        # 5. Rate Limit
        # Send at roughly 20 FPS
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nStopping...")
    sock.close()
except Exception as e:
    print(f"Error: {e}")
    sock.close()