# Sound Simulation

A real-time sound classification and visualization system that combines acoustic direction-of-arrival (DOA) tracking with wave physics simulation. The system classifies sounds into three categories (whistles, snaps, and speech) and renders them as interactive particles on a 2D wave propagation grid.

![Sound Classifier Demo](https://img.shields.io/badge/Python-3.7+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features

- **Real-time Sound Classification**: Distinguishes between whistles, snaps, speech, and silence using spectral analysis
- **Direction-of-Arrival Tracking**: Integrates with ReSpeaker 6-mic array for spatial audio localization
- **2D Wave Physics Simulation**: Visualizes sound propagation using the wave equation with damping
- **Frequency Analysis Visualization**: Real-time winding machine display for dominant frequency extraction
- **Color-coded Particle System**: Visual representation of classified sounds (Red=Snap, Cyan=Whistle, Green=Speech)

## Table of Contents

- [Installation](#installation)
- [Hardware Requirements](#hardware-requirements)
- [Usage](#usage)
- [Technical Details](#technical-details)
- [Classification Algorithm](#classification-algorithm)
- [Wave Physics](#wave-physics)
- [Credits](#credits)
- [License](#license)

## Installation

### Prerequisites

```bash
pip install numpy sounddevice scipy matplotlib pyusb
```

### ReSpeaker Setup

This project uses the ReSpeaker USB 6-mic array for directional audio capture. The `tuning.py` library is adapted from the official ReSpeaker repository.

**Original Source**: [Anplus/usb_4_mic_array](https://github.com/Anplus/usb_4_mic_array)  
**Reference Implementation**: [DOA3.py](https://github.com/Anplus/usb_4_mic_array/blob/master/DOA3.py)

#### Important Note
The original `tuning.py` contains a bug in the `read()` method. You must update line 127:

```python
# INCORRECT (original):
response = struct.unpack(b'ii', response.toString())

# CORRECT (required):
response = struct.unpack(b'ii', response.tobytes())
```

### Clone Repository

```bash
git clone https://github.com/yourusername/sound-simulation.git
cd sound-simulation
```

## Hardware Requirements

### Supported Devices

- **ReSpeaker USB 6-Mic Circular Array** (VID: 0x2886, PID: 0x0018)
- Any USB audio interface with 6+ channels (with reduced DOA functionality)

### Finding Your Audio Device

```python
import sounddevice as sd
print(sd.query_devices())
```

Update `DEVICE_INDEX` in `sound_classifier.py` to match your hardware.

## Usage

### Basic Execution

```bash
python sound_classifier.py
```

### Testing DOA Only

```bash
python DOA3.py
```

### Configuration

Edit the following parameters in `sound_classifier.py`:

```python
SAMPLE_RATE = 16000      # Audio sampling rate (Hz)
BLOCKSIZE = 2048         # FFT window size
NUM_MICS = 6             # Number of microphone channels
DEVICE_INDEX = 1         # Audio device index

# Physics parameters
WAVE_SPEED = 0.45        # Wave propagation speed
DAMPING = 0.98           # Energy decay factor
GRID_W = 200             # Grid width (pixels)
GRID_H = 150             # Grid height (pixels)
```

## Technical Details

### Classification Algorithm

The system uses a multi-metric decision tree based on spectral analysis:

#### 1. **Spectral Flatness (Wiener Entropy)**

$$
\text{Flatness} = \frac{\text{GM}(S)}{\text{AM}(S)} = \frac{\sqrt[N]{\prod_{i=1}^{N} S_i}}{\frac{1}{N}\sum_{i=1}^{N} S_i}
$$

Where:
- $S_i$ = Magnitude spectrum at frequency bin $i$
- $N$ = Number of frequency bins
- GM = Geometric Mean
- AM = Arithmetic Mean

**Interpretation**:
- Low flatness (≈ 0.0) → Pure tone (whistle)
- High flatness (≈ 1.0) → Broadband noise (snap)

#### 2. **Peak Ratio**

$$
\text{Peak Ratio} = \frac{\max(S)}{\text{mean}(S)}
$$

Measures how much energy is concentrated at the dominant frequency.

#### 3. **Classification Rules**

```python
if flatness > 0.5 and volume > 10.0:
    return 'SNAP'
    
elif flatness < 0.15 and peak_ratio > 10.0 and 800 < peak_freq < 3500:
    return 'WHISTLE'
    
else:
    return 'SPEECH'
```

### Wave Physics

The simulation solves the 2D damped wave equation:

$$
\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u - \gamma \frac{\partial u}{\partial t}
$$

Where:
- $u(x,y,t)$ = Wave amplitude
- $c$ = Wave speed (`WAVE_SPEED`)
- $\gamma$ = Damping coefficient (`1 - DAMPING`)
- $\nabla^2$ = Laplacian operator

#### Discrete Implementation

Using finite differences with centered spatial derivatives:

$$
u_{i,j}^{n+1} = 2u_{i,j}^{n} - u_{i,j}^{n-1} + c^2 \Delta t^2 \nabla^2 u_{i,j}^{n}
$$

$$
\nabla^2 u_{i,j} \approx u_{i-1,j} + u_{i+1,j} + u_{i,j-1} + u_{i,j+1} - 4u_{i,j}
$$

With damping applied:

$$
u^{n+1} \leftarrow \alpha \cdot u^{n+1}
$$

Where $\alpha$ = `DAMPING` (default: 0.98)

### Direction-of-Arrival (DOA)

The ReSpeaker array provides DOA angles (0-359°) using time-difference-of-arrival (TDOA) algorithms. Particles spawn at positions calculated from the DOA:

$$
\begin{align}
x &= \frac{W}{2} + r \cos(\theta) \\
y &= \frac{H}{2} + r \sin(\theta)
\end{align}
$$

Where:
- $\theta$ = DOA angle (radians)
- $r$ = Spawn radius (default: 50 pixels)
- $W, H$ = Grid dimensions

### Frequency Visualization (Winding Machine)

The winding machine maps time-domain signals to a spiral in the complex plane:

$$
z(t) = A(t) \cdot e^{-i 2\pi f t}
$$

Where:
- $A(t)$ = Amplitude envelope
- $f$ = Winding frequency

The center-of-mass of this spiral reveals the dominant frequency through constructive interference.

## File Structure

```
sound-simulation/
├── sound_classifier.py    # Main application
├── DOA3.py               # DOA testing script
├── tuning.py             # ReSpeaker control library (modified)
└── README.md             # This file
```

## Credits

### Hardware Libraries

- **ReSpeaker Tuning Library**: Adapted from [Anplus/usb_4_mic_array](https://github.com/Anplus/usb_4_mic_array)
- **DOA Reference**: [DOA3.py](https://github.com/Anplus/usb_4_mic_array/blob/master/DOA3.py)

### Software Dependencies

- [NumPy](https://numpy.org/) - Numerical computing
- [SciPy](https://scipy.org/) - FFT and signal processing
- [Matplotlib](https://matplotlib.org/) - Visualization
- [sounddevice](https://python-sounddevice.readthedocs.io/) - Audio I/O
- [PyUSB](https://github.com/pyusb/pyusb) - USB device communication

## Troubleshooting

### No Audio Device Found

```bash
# List available devices
python -c "import sounddevice as sd; print(sd.query_devices())"
```

Update `DEVICE_INDEX` in the code.

### ReSpeaker Not Detected

```bash
# Check USB connection
lsusb | grep 2886:0018

# Verify permissions (Linux)
sudo usermod -a -G dialout $USER
```

### ImportError: tuning.py

Ensure `tuning.py` is in the same directory and contains the `tobytes()` fix.

### Poor Classification Performance

Adjust sensitivity thresholds:

```python
# In classify_sound() function
vol < 1.0        # Silence threshold (increase for noisy environments)
flatness > 0.5   # Snap detection (decrease for stricter classification)
flatness < 0.15  # Whistle purity (increase to catch weaker whistles)
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## Acknowledgments

Special thanks to the ReSpeaker team and the open-source audio processing community for making spatial audio accessible.

---

**Author**: Gilbert Baraka  
**Repository**: https://github.com/yourusername/sound-simulation  
**Issues**: https://github.com/yourusername/sound-simulation/issues
