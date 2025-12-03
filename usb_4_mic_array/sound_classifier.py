import numpy as np
import sounddevice as sd
from collections import deque
from scipy.fftpack import rfft, rfftfreq
from scipy.stats import gmean
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
import usb.core
import usb.util
import math

# Try to import Tuning (ReSpeaker)
try:
    from tuning import Tuning
    HAS_TUNING_LIB = True
except ImportError:
    print("WARNING: 'tuning.py' not found. DOA features disabled.")
    print("If tuning exist, then then update it to ---> response = struct.unpack(b'ii', response.tobytes()) NOT toSting()")
    HAS_TUNING_LIB = False

# ---------------------------
# User Settings
# ---------------------------
SAMPLE_RATE = 16000
BLOCKSIZE = 2048   # Larger block helps frequency resolution
NUM_MICS = 6       # ReSpeaker Index 1
DEVICE_INDEX = 1   

# Physics Constants
WAVE_SPEED = 0.45
DAMPING = 0.98
GRID_W = 200
GRID_H = 150
PHYSICS_STEPS_PER_FRAME = 6

# Visual Tuning
DOT_DECAY = 0.08
MAX_DOTS = 80
GLOW_SIZE = 80

# ---------------------------
# Sound Classification Logic
# ---------------------------
def classify_sound(audio_signal, sample_rate):
    """
    Returns: 'WHISTLE', 'SNAP', 'SPEECH', or 'SILENCE'
    """
    # 1. Silence Check
    vol = np.linalg.norm(audio_signal)
    if vol < 1.0: # Adjust based on your mic sensitivity
        return 'SILENCE', vol

    # 2. Frequency Analysis
    # Apply window to reduce spectral leakage
    window = np.hamming(len(audio_signal))
    spectrum = np.abs(rfft(audio_signal * window))
    freqs = rfftfreq(len(audio_signal), 1/sample_rate)
    
    # Avoid divide by zero in geometric mean
    spectrum[spectrum == 0] = 1e-10
    
    # METRIC 1: Spectral Flatness (Wiener Entropy)
    # Low (near 0.0) = Pure Tone (Whistle)
    # High (near 1.0) = Noisy/Impulse (Snap)
    flatness = gmean(spectrum) / np.mean(spectrum)
    
    # METRIC 2: Dominant Frequency
    peak_idx = np.argmax(spectrum)
    peak_freq = freqs[peak_idx]
    
    # METRIC 3: Energy Concentration (Peak Ratio)
    # How much stronger is the peak than the average?
    peak_ratio = spectrum[peak_idx] / np.mean(spectrum)

    # --- CLASSIFICATION TREE ---
    
    # SNAP DETECTION
    # Snaps are sudden (high vol), broadband (high flatness), and often have no dominant pitch
    if flatness > 0.5 and vol > 10.0: 
        return 'SNAP', vol

    # WHISTLE DETECTION
    # Whistles are pure tones (low flatness), high peak ratio, in specific range (800-3500Hz)
    if (flatness < 0.15 and 
        peak_ratio > 10.0 and 
        800 < peak_freq < 3500):
        return 'WHISTLE', vol

    # SPEECH / GENERAL
    return 'SPEECH', vol

# ---------------------------
# Hardware Setup
# ---------------------------
mic_tuning = None
if HAS_TUNING_LIB:
    dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    if dev:
        mic_tuning = Tuning(dev)
        print(f"ReSpeaker Found. Initial Direction: {mic_tuning.direction}")
    else:
        print("ReSpeaker Not Found.")

# ---------------------------
# Audio Processing
# ---------------------------
audio_queue = deque(maxlen=2)

def audio_callback(indata, frames, time_info, status):
    if status: print(status)
    mono = np.mean(indata, axis=1)
    audio_queue.append(mono.copy())

# ---------------------------
# Wave Physics
# ---------------------------
u = np.zeros((GRID_H, GRID_W), dtype=np.float32)
u_prev = np.zeros((GRID_H, GRID_W), dtype=np.float32)
walls_grid = np.zeros((GRID_H, GRID_W), dtype=bool)
walls_grid[50:100, 70] = 1
walls_grid[50:100, 130] = 1

def step_wave_equation():
    global u, u_prev
    laplacian = (u[0:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, 0:-2] + u[1:-1, 2:] - 4*u[1:-1, 1:-1])
    u_new = np.zeros_like(u)
    u_new[1:-1, 1:-1] = 2*u[1:-1, 1:-1] - u_prev[1:-1, 1:-1] + (WAVE_SPEED**2)*laplacian
    u_new *= DAMPING
    u_new[walls_grid] = 0
    u_prev[:] = u[:]
    u[:] = u_new[:]

# ---------------------------
# Visualization Setup
# ---------------------------
fig = plt.figure(figsize=(14, 7), facecolor='#1e1e1e')
gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 0.8])

# Wave Panel
ax1 = fig.add_subplot(gs[0])
ax1.set_title("Sound Classifier (Snap=Red, Whistle=Cyan)", color='white')
ax1.axis('off')
img_wave = ax1.imshow(u, cmap='RdBu', vmin=-0.5, vmax=0.5, origin='lower', interpolation='bicubic')
wall_overlay = np.ma.masked_where(walls_grid == 0, walls_grid)
ax1.imshow(wall_overlay, cmap='gray', vmin=0, vmax=1, origin='lower', alpha=0.6)

# Particles: [x, y, life, type_id]
# Type ID: 0=Speech(Green), 1=Whistle(Cyan), 2=Snap(Red)
particle_scatter = ax1.scatter([], [], c=[], s=GLOW_SIZE, edgecolors='none', zorder=10)
particles = []

# Freq Panel
ax2 = fig.add_subplot(gs[1])
ax2.set_title("Frequency Analysis", color='white')
ax2.set_facecolor('black')
ax2.axis('off')
winding_line, = ax2.plot([], [], color='cyan', alpha=0.5, lw=0.8)
center_dot, = ax2.plot([], [], 'o', color='red', ms=8)
freq_text = ax2.text(0.05, 0.95, "", transform=ax2.transAxes, color='white', fontsize=14)
type_text = ax2.text(0.05, 0.85, "Ready...", transform=ax2.transAxes, color='lime', fontsize=16)

stream = sd.InputStream(samplerate=SAMPLE_RATE, blocksize=BLOCKSIZE,
                        channels=NUM_MICS, callback=audio_callback,
                        device=DEVICE_INDEX)
stream.start()

def update(frame):
    global particles
    
    # 1. Process Audio & Classification
    sound_type = 'SILENCE'
    vol = 0
    if len(audio_queue) > 0:
        audio = audio_queue[-1]
        sound_type, vol = classify_sound(audio, SAMPLE_RATE)
        
        # Determine Color ID based on sound type
        color_id = 0 # Green (Default/Speech)
        if sound_type == 'WHISTLE': color_id = 1 # Cyan
        elif sound_type == 'SNAP':   color_id = 2 # Red

        # Update Text
        if sound_type != 'SILENCE':
            type_text.set_text(f"Detected: {sound_type}")
            if sound_type == 'SNAP': type_text.set_color('red')
            elif sound_type == 'WHISTLE': type_text.set_color('cyan')
            else: type_text.set_color('lime')

        # 2. Spawn Particles
        if mic_tuning and sound_type != 'SILENCE':
            try:
                doa = mic_tuning.direction
                rad = math.radians(doa)
                radius = 50 
                dx, dy = int(radius * math.cos(rad)), int(radius * math.sin(rad))
                tx, ty = np.clip(GRID_W//2 + dx, 2, GRID_W-3), np.clip(GRID_H//2 + dy, 2, GRID_H-3)
                
                # Add particle [x, y, life, color_id]
                particles.append([tx, ty, 1.0, color_id])
                
                # Inject Physics Wave
                # Snaps make bigger splashes!
                splash_gain = 2.0 if sound_type == 'SNAP' else 0.8
                u_prev[ty, tx] += vol * splash_gain
                
            except: pass

        # Winding Machine
        dom_freq = 0
        if sound_type != 'SILENCE':
            # Simplified frequency for winding
            spectrum = np.abs(rfft(audio))
            freqs = rfftfreq(len(audio), 1/SAMPLE_RATE)
            dom_freq = freqs[np.argmax(spectrum)]
            
            # Winding math
            n = len(audio)
            t = np.linspace(0, n/SAMPLE_RATE, n)
            angle = -2 * np.pi * dom_freq * t
            radius_viz = audio * 5.0 + 1.0
            xs = radius_viz * np.cos(angle)
            ys = radius_viz * np.sin(angle)
            
            winding_line.set_data(xs, ys)
            center_dot.set_data([np.mean(xs)], [np.mean(ys)])
            freq_text.set_text(f"{dom_freq:.0f} Hz")
            ax2.set_xlim(-5, 5); ax2.set_ylim(-5, 5)

    # 3. Update Particles
    for p in particles:
        p[2] -= DOT_DECAY # Decay life
    particles = [p for p in particles if p[2] > 0]

    if particles:
        data = np.array(particles)
        particle_scatter.set_offsets(data[:, :2])
        
        # Color Map Logic
        # We build an RGBA array for every particle
        rgba = np.zeros((len(particles), 4))
        for i, p in enumerate(particles):
            cid = p[3]
            life = p[2]
            if cid == 2:   rgba[i] = [1.0, 0.2, 0.2, life] # RED (Snap)
            elif cid == 1: rgba[i] = [0.0, 1.0, 1.0, life] # CYAN (Whistle)
            else:          rgba[i] = [0.2, 1.0, 0.2, life] # GREEN (Speech)
            
        particle_scatter.set_color(rgba)
    else:
        particle_scatter.set_offsets(np.empty((0, 2)))

    # 4. Physics Stepping
    for _ in range(PHYSICS_STEPS_PER_FRAME):
        step_wave_equation()
        
    img_wave.set_data(u)
    return [img_wave, winding_line, center_dot, freq_text, type_text, particle_scatter]

ani = FuncAnimation(fig, update, interval=1, blit=False)
plt.show()
stream.stop()
stream.close()