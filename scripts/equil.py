import argparse
import dwfpy as dwf
import matplotlib.pyplot as plt
import time
from collections import defaultdict
from util.measurements import get_measurements
from util.append_data import append_data

FREQ = 117e3  # 117 kHz
TAU = 120     # 120 second thermal time constant

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

parser = argparse.ArgumentParser(description='Drone Measurement Script')
parser.add_argument('-t', '--taus', type=float, default=5.0, help='Number of time constants to run the script for.')
parser.add_argument('-o', '--output', type=str, default=None, help='Output CSV file to save results.')

args = parser.parse_args()
output_file = args.output
t_taus = args.taus
t_sec = t_taus * TAU

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

    input('Press Enter to start the system: ')

    pattern = device.digital_output
    pattern[0].setup_clock(frequency=FREQ, configure=True, start=True)
    device.digital_io[1].setup(enabled=True, state=True, configure=True)

    logic = device.digital_input
    logic.setup_edge_trigger(channel=FORCE_PIN, edge='rising')

    start_time = time.time()
    timeout = start_time + t_sec
    data = defaultdict(list)

    while time.time() < timeout:
        scope.single(sample_rate=SCOPE_SAMPLE_RATE, buffer_size=SCOPE_BUFFER_SIZE, configure=True, start=True)
        ch1 = scope[0].get_data() * CH1_ATTEN
        ch2 = scope[1].get_data() * CH2_ATTEN
        ch3 = scope[2].get_data() * CH3_ATTEN
        ch4 = scope[3].get_data() * CH4_ATTEN

        logic.single(sample_rate=PIN_SAMPLE_RATE, buffer_size=PIN_BUFFER_SIZE, configure=True, start=True)
        pin_data = logic.get_data()

        results = get_measurements(ch1, ch2, ch3, ch4, SCOPE_SAMPLE_RATE, pin_data, FORCE_PIN)
        elapsed = time.time() - start_time
        results['Time (s)'] = elapsed

        if output_file:
            append_data(output_file, results)
        
        for key, value in results.items():
            data[key].append(value)

        timestamp = time.strftime('%H:%M:%S')
        force_mn = results['RX Force (mN)']
        force_g = force_mn / 9.81
        print(f'[{timestamp}] Force: {force_mn:6.2f} mN ({force_g:5.1f}g)')

        time.sleep(5)


    print(f'\n--> Equilibrium state reached.')
    if output_file:
        print(f"--> Measurements saved to {output_file}")


    print('Close the figure to stop the program.')
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    plt.suptitle(f"System Measurements Over Time")

    axes[0, 0].plot(data['Time (s)'], data['RX Force (mN)'], color='tab:blue')
    axes[0, 0].set_ylabel('RX Force (mN)')
    axes[0, 0].set_title('RX Force')
    axes[0, 0].grid(True)

    axes[0, 1].plot(data['Time (s)'], data['RX Voltage RMS (V)'], color='tab:orange')
    axes[0, 1].set_ylabel('RX Voltage RMS (V)')
    axes[0, 1].set_title('RX Voltage RMS')
    axes[0, 1].grid(True)

    axes[1, 0].plot(data['Time (s)'], data['TX Voltage Peak-to-Peak (V)'], color='tab:green')
    axes[1, 0].set_ylabel('TX Voltage Peak-to-Peak (V)')
    axes[1, 0].set_title('TX Voltage Peak-to-Peak')
    axes[1, 0].grid(True)

    axes[1, 1].plot(data['Time (s)'], data['TX Current Peak-to-Peak (A)'], color='tab:purple')
    axes[1, 1].set_ylabel('TX Current Peak-to-Peak (A)')
    axes[1, 1].set_title('TX Current Peak-to-Peak')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

    device.digital_io[0].output_state = False
    device.digital_io[1].output_state = False
    device.close()