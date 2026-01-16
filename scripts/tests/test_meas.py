import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from measurements import get_measurements

# --- 1. Simulation Settings ---
SAMPLE_RATE = 20e6      # 20 MHz resolution
DURATION    = 0.0001    # 100 microseconds (enough for ~10 cycles of 117kHz)
FREQ        = 117e3     # 117 kHz
t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION))

# --- 2. Generate Synthetic Data ---
print("Generating 4 Channels of Test Data...")

# Noise Generator
def add_noise(signal, level):
    return signal + np.random.normal(0, level, len(t))

# Ch1: Power Supply (10V DC + Sag + Noise)
# We add a slow sine wave (5kHz) to simulate "Sag/Ripple"
sag = 0.2 * np.sin(2 * np.pi * 5000 * t) 
ch1_raw = 10.0 + sag 
ch1 = add_noise(ch1_raw, level=0.05) # +/- 50mV noise

# Ch2: Current Sine (12A Amplitude)
# Note: 12A Amplitude = 8.48A RMS
ch2_raw = 12.0 * np.sin(2 * np.pi * FREQ * t)
ch2 = add_noise(ch2_raw, level=0.2) # +/- 0.2A noise

# Ch3: Voltage Sine (700V Amplitude)
# Note: 700V Amplitude = 495V RMS
ch3_raw = 700.0 * np.sin(2 * np.pi * FREQ * t + 0.5) # Phase shift
ch3 = add_noise(ch3_raw, level=10.0) # +/- 10V noise

# Ch4: Rectified Sine (5V Amplitude)
# abs() creates the rectification
ch4_raw = np.abs(5.0 * np.sin(2 * np.pi * FREQ * t))
ch4 = add_noise(ch4_raw, level=0.1) # +/- 100mV noise

# --- 4. Run Analysis ---
results = get_measurements(ch1, ch2, ch3, ch4, SAMPLE_RATE)

# --- 5. Display Results ---
print("\n--- MEASUREMENT RESULTS ---")
df = pd.DataFrame([results]) # Transpose for easier reading
print(df.T)

# # --- 6. Plotting (The Verification) ---
# fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
# plt.suptitle(f"Simulated Inputs ({FREQ/1000:.1f} kHz)")

# # Plot Ch1 (DC)
# axes[0, 0].plot(t*1e6, ch1, color='tab:blue')
# axes[0, 0].set_ylabel('Supply (V)')
# axes[0, 0].set_title(f"Ch1: DC Supply (Mean={results['Power Supply Average (V)']:.2f}V)")
# axes[0, 0].grid(True)

# # Plot Ch2 (Current)
# axes[0, 1].plot(t*1e6, ch2, color='tab:green')
# axes[0, 1].set_ylabel('Current (A)')
# axes[0, 1].set_title(f"Ch2: Current (RMS={results['TX Current RMS (A)']:.2f}A, Amp={results['TX Current Peak-to-Peak (A)']:.2f}A)")
# axes[0, 1].grid(True)

# # Plot Ch3 (HV Voltage)
# axes[1, 0].plot(t*1e6, ch3, color='tab:orange')
# axes[1, 0].set_ylabel('Voltage (V)')
# axes[1, 0].set_title(f"Ch3: HV Voltage (RMS={results['TX Voltage RMS (V)']:.2f}V)")
# axes[1, 0].grid(True)

# # Plot Ch4 (Rectified)
# axes[1, 1].plot(t*1e6, ch4, color='tab:purple')
# axes[1, 1].set_ylabel('Rectified (V)')
# axes[1, 1].set_title(f"Ch4: Rectified (Avg={results['RX Voltage Average (V)']:.2f}V, TrueRMS={results['RX Voltage RMS (V)']:.2f}V)")
# axes[1, 1].set_xlabel('Time (microseconds)')
# axes[1, 1].grid(True)

# plt.tight_layout()
# plt.show()

filename = "simulated_measurements.csv"
# df.to_csv(filename, index=False)
df.to_csv(filename, mode='a', header=not pd.io.common.file_exists(filename), index=False)