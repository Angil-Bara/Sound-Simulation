from tuning import Tuning
import usb.core
import usb.util
import time

# Find the USB device by Vendor ID (VID) and Product ID (PID)
dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)

if dev:
    # Initialize the Tuning object
    mic_tuning = Tuning(dev)
    
    # Print the initial direction
    print(f"Initial Direction: {mic_tuning.direction}")
    
    # Main loop to continuously print the direction
    while True:
        try:
            # Python 3 print function syntax
            print(f"Current Direction: {mic_tuning.direction}")
            time.sleep(1)
            
        # Updated exception handling (Python 3 requires 'as e' for accessing the exception object)
        except KeyboardInterrupt:
            print("\nExiting loop...")
            break
            
else:
    print("Error: USB device (VID=0x2886, PID=0x0018) not found.")