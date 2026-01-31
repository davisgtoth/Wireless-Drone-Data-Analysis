import dwfpy as dwf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from util.measurements import get_measurements

FREQ = 117e3  # 117 kHz

CH1_ATTEN = 10      # Power Supply Voltage
CH2_ATTEN = 1/50e-3 # Current Probe 50 mV/A
CH3_ATTEN = 500     # TX Coil Voltage
CH4_ATTEN = 10      # RX Coil Voltage

SCOPE_SAMPLE_RATE = 2e7 
SCOPE_BUFFER_SIZE = 1000 

FORCE_PIN = 2
FORCE_FREQ = 1e3
PIN_SAMPLE_RATE = 1e6
NUM_PERIODS = 2
PIN_BUFFER_SIZE = int(PIN_SAMPLE_RATE / FORCE_FREQ * NUM_PERIODS) 

with dwf.Device() as device:
    print(f"Device {device.name} {device.serial_number} opened successfully.")
    
    scope = device.analog_input
    scope[0].setup(range=10.0, offset=0.0) 
    scope[1].setup(range=10.0, offset=0.0) 
    scope[2].setup(range=10.0, offset=0.0) 
    scope[3].setup(range=10.0, offset=0.0) 

    scope.setup_edge_trigger(channel=0, mode='auto')
    scope.setup_edge_trigger(channel=1, mode='auto')
    scope.setup_edge_trigger(channel=2, mode='auto')
    scope.setup_edge_trigger(channel=3, mode='auto')

    input('Press Enter to start: ')
    
    pattern = device.digital_output
    pattern[0].setup_clock(frequency=FREQ, configure=True, start=True)
    device.digital_io[1].setup(enabled=True, state=True, configure=True)

    logic = device.digital_input
    logic.setup_edge_trigger(channel=FORCE_PIN, edge='rising')

    # input('Press Enter to to change frequency to 116 kHz: ')
    # pattern[0].setup_clock(frequency=116e3, configure=True, start=True)

    input('Press Enter to gather data then stop: ')

    scope.single(sample_rate=SCOPE_SAMPLE_RATE, buffer_size=SCOPE_BUFFER_SIZE, configure=True, start=True)
    ch1 = scope[0].get_data() * CH1_ATTEN
    ch2 = scope[1].get_data() * CH2_ATTEN
    ch3 = scope[2].get_data() * CH3_ATTEN
    ch4 = scope[3].get_data() * CH4_ATTEN

    logic.single(sample_rate=PIN_SAMPLE_RATE, buffer_size=PIN_BUFFER_SIZE, configure=True, start=True)
    pin_data = logic.get_data()

    results = get_measurements(ch1, ch2, ch3, ch4, SCOPE_SAMPLE_RATE, pin_data, FORCE_PIN)

    device.digital_io[0].output_state = False
    device.digital_io[1].output_state = False
    device.close()


print("\n--- MEASUREMENT RESULTS ---")
df = pd.DataFrame([results]).T 
print(df)


fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
plt.suptitle(f"Scope Capture")
t = [i / SCOPE_SAMPLE_RATE * 1e6 for i in range(len(ch1))]

axes[0, 0].plot(t, ch1, color='tab:blue')
axes[0, 0].set_ylabel('Ch1 Voltage (V)')
axes[0, 0].set_title('Channel 1: Power Supply Voltage')
axes[0, 0].grid(True)

axes[0, 1].plot(t, ch2, color='tab:orange')
axes[0, 1].set_ylabel('Ch2 Current (A)')
axes[0, 1].set_title('Channel 2: TX Current')
axes[0, 1].grid(True)

axes[1, 0].plot(t, ch3, color='tab:green')
axes[1, 0].set_ylabel('Ch3 Voltage (V)')
axes[1, 0].set_title('Channel 3: TX Voltage')
axes[1, 0].grid(True)

axes[1, 1].plot(t, ch4, color='tab:purple')
axes[1, 1].set_ylabel('Ch4 Voltage (V)')
axes[1, 1].set_title('Channel 4: RX Voltage')
axes[1, 1].set_xlabel('Time (microseconds)')
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

time_axis = np.linspace(0, PIN_BUFFER_SIZE/PIN_SAMPLE_RATE, len(pin_data))
plt.figure(figsize=(10, 4))
plt.step(time_axis * 1000, (pin_data >> FORCE_PIN) & 1, where='post')
plt.title(f'Force Pin Signal - Mass in Grams: {results["RX Force (mN)"]/9.81:.2f} g')
plt.xlabel('Time (milliseconds)')
plt.ylabel('Digital Level')
plt.grid(True)
plt.show()