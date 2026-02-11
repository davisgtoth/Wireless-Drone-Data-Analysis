import argparse
import dwfpy as dwf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from util.measurements import get_measurements
from util.append_data import append_data

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

parser = argparse.ArgumentParser(description='Drone Measurement Script')
parser.add_argument('-t', '--time', type=float, default=5.0, help='Duration to run the script in minutes.')
parser.add_argument('-f', '--force', action='store_true', help='Include force measurement from digital pin.')
parser.add_argument('-v', '--voltage', action='store_true', help='Include voltage measurements from scope channels.')
parser.add_argument('-s', '--sweep', action='store_true', help='Perform frequency sweeps over the specified time interval.')
parser.add_argument('-o', '--output', type=str, required=True, help='Output CSV file to save results.')

args = parser.parse_args()
output_file = args.output
t_min = args.time
t_sec = t_min * 60

if not (args.force or args.voltage or args.sweep):
    print("No measurement type specified. Use -f, -v, or -s to select measurements.")
    exit(1)

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

    input('Press Enter to start the system and data collection: ')

    pattern = device.digital_output
    pattern[0].setup_clock(frequency=FREQ, configure=True, start=True)
    device.digital_io[1].setup(enabled=True, state=True, configure=True)

    logic = device.digital_input
    logic.setup_edge_trigger(channel=FORCE_PIN, edge='rising')

    time.sleep(1) 

    start_time = time.time()
    timeout = start_time + t_sec

    while time.time() < timeout:
        if args.force:
            logic.single(sample_rate=PIN_SAMPLE_RATE, buffer_size=PIN_BUFFER_SIZE, configure=True, start=True)
            pin_data = logic.get_data()

            elapsed = time.time() - start_time
            signal = (pin_data >> FORCE_PIN) & 1
            duty_cycle = np.sum(signal) / len(signal)

            data = {
                'Time (s)': elapsed,
                'RX Force (mN)': duty_cycle*50 * 9.81  # duty cycle * 50 = mass in grams
            }
            append_data(output_file, data)


        elif args.voltage:
            scope.single(sample_rate=SCOPE_SAMPLE_RATE, buffer_size=SCOPE_BUFFER_SIZE, configure=True, start=True)
            ch1 = scope[0].get_data() * CH1_ATTEN
            ch2 = scope[1].get_data() * CH2_ATTEN
            ch3 = scope[2].get_data() * CH3_ATTEN
            ch4 = scope[3].get_data() * CH4_ATTEN

            logic.single(sample_rate=PIN_SAMPLE_RATE, buffer_size=PIN_BUFFER_SIZE, configure=True, start=True)
            pin_data = logic.get_data()

            results = get_measurements(ch1, ch2, ch3, ch4, SCOPE_SAMPLE_RATE, pin_data, FORCE_PIN)
            elapsed = time.time() - start_time
            data = {
                'Time (s)': elapsed,
                **results
            }
            append_data(output_file, data)


        elif args.sweep:
            start_freq = 115e3
            stop_freq = 120e3
            step_freq = 0.1e3
            freq_list = np.arange(start_freq, stop_freq + step_freq, step_freq)

            # pattern[0].setup_clock(frequency=freq_list[0], configure=True, start=True)
            # time.sleep(1)

            pbar = tqdm(freq_list, desc='Sweeping', unit='kHz', ncols=100)
            freq_start_delT = time.time() - start_time

            for freq in pbar:
                pattern[0].setup_clock(frequency=freq, configure=True, start=True)
                # time.sleep(2)
                time.sleep(0.05)

                scope.single(sample_rate=SCOPE_SAMPLE_RATE, buffer_size=SCOPE_BUFFER_SIZE, configure=True, start=True)
                ch1 = scope[0].get_data() * CH1_ATTEN
                ch2 = scope[1].get_data() * CH2_ATTEN
                ch3 = scope[2].get_data() * CH3_ATTEN
                ch4 = scope[3].get_data() * CH4_ATTEN

                # logic.single(sample_rate=PIN_SAMPLE_RATE, buffer_size=PIN_BUFFER_SIZE, configure=True, start=True)
                # pin_data = logic.get_data()

                # results = get_measurements(ch1, ch2, ch3, ch4, SCOPE_SAMPLE_RATE, pin_data, FORCE_PIN)
                results = get_measurements(ch1, ch2, ch3, ch4, SCOPE_SAMPLE_RATE)
                data = {
                    'Start Time (s)': freq_start_delT,
                    'Driving Frequency (Hz)': freq,
                    **results
                }
                append_data(output_file, data)

                # force = results['RX Force (mN)']
                # pbar.set_postfix(f'Freq={freq/1e3:.1f}kHz')
                tx_rms = results['TX Voltage RMS (V)']
                pbar.set_postfix_str(f'Freq={freq/1e3:.1f}k | TX_RMS={tx_rms:.2f}V')


        if not args.sweep:
            timestamp = time.strftime('%H:%M:%S')
            force_mn = data['RX Force (mN)']
            force_g = force_mn / 9.81
            print(f'[{timestamp}] Force: {force_mn:6.2f} mN ({force_g:5.1f}g)')

        # time.sleep(15)
        pattern[0].setup_clock(frequency=FREQ, configure=True, start=True)
        # time.sleep(5)
        time.sleep(30)

    print(f"\n--> Measurement complete. Results saved to {output_file}")

    device.digital_io[0].output_state = False
    device.digital_io[1].output_state = False
    device.close()


if args.force:
    df = pd.read_csv(output_file)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['Time (s)'], df['RX Force (mN)'], '-o')
    plt.title('RX Force vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('RX Force (mN)')
    plt.grid(True)
    plt.show()


elif args.voltage:
    df = pd.read_csv(output_file)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    plt.suptitle(f"System Measurements Over Time")

    axes[0, 0].plot(df['Time (s)'], df['RX Force (mN)'], color='tab:blue')
    axes[0, 0].set_ylabel('RX Force (mN)')
    axes[0, 0].set_title('RX Force')
    axes[0, 0].grid(True)

    axes[0, 1].plot(df['Time (s)'], df['RX Voltage RMS (V)'], color='tab:orange')
    axes[0, 1].set_ylabel('RX Voltage RMS (V)')
    axes[0, 1].set_title('RX Voltage RMS')
    axes[0, 1].grid(True)

    axes[1, 0].plot(df['Time (s)'], df['TX Voltage Peak-to-Peak (V)'], color='tab:green')
    axes[1, 0].set_ylabel('TX Voltage Peak-to-Peak (V)')
    axes[1, 0].set_title('TX Voltage Peak-to-Peak')
    axes[1, 0].grid(True)

    axes[1, 1].plot(df['Time (s)'], df['TX Current Peak-to-Peak (A)'], color='tab:purple')
    axes[1, 1].set_ylabel('TX Current Peak-to-Peak (A)')
    axes[1, 1].set_title('TX Current Peak-to-Peak')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()