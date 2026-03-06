import argparse
import dwfpy as dwf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from util.measurements import get_measurements
from util.append_data import append_data

# FREQ = 119e3  # 119 kHz
FREQ = 118e3  # 118 kHz

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
parser.add_argument('-f', '--force', action='store_true', help='Include force measurement from digital pin.')
parser.add_argument('-o', '--output', type=str, required=True, help='Output CSV file to save results.')

args = parser.parse_args()
output_file = args.output

with dwf.Device() as device:
    print(f"--> Connected to: {device.name} {device.serial_number}\n")
    
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

    try:
        while True:
            user_in = input('Press Enter to gather data (or "q"): ').strip()

            if user_in.lower() == 'q':
                print('Exiting...')
                break
            
            print('gathering data...')

            scope.single(sample_rate=SCOPE_SAMPLE_RATE, buffer_size=SCOPE_BUFFER_SIZE, configure=True, start=True)
            ch1 = scope[0].get_data() * CH1_ATTEN
            ch2 = scope[1].get_data() * CH2_ATTEN
            ch3 = scope[2].get_data() * CH3_ATTEN
            ch4 = scope[3].get_data() * CH4_ATTEN

            if args.force:
                logic.single(sample_rate=PIN_SAMPLE_RATE, buffer_size=PIN_BUFFER_SIZE, configure=True, start=True)
                pin_data = logic.get_data()
            else:
                pin_data = None

            results = get_measurements(ch1, ch2, ch3, ch4, SCOPE_SAMPLE_RATE, pin_data, FORCE_PIN)

            data = {
                'Driving Frequency (Hz)': FREQ,
                **results
            }
            append_data(output_file, data)

    except KeyboardInterrupt:
        print('Force quit detected. Exiting...')

    finally:
        device.digital_io[0].output_state = False
        device.digital_io[1].output_state = False
        device.close()