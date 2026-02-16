import argparse
import numpy as np
import dwfpy as dwf
from util.measurements import get_measurements
from util.append_data import append_data

parser = argparse.ArgumentParser(description="Perform a coordinate sweep and log measurements.")
parser.add_argument('-f', '--freq', type=float, default=117.0, help='Frequency in kHz for the measurement.')
parser.add_argument('--force', action='store_true', help='Include force measurement from digital pin.')
parser.add_argument('-c', '--sweep', type=str, required=True, choices=['r', 'theta', 'z'], 
                    help='Coordinate varied in the measurement (r, theta, or z).')
parser.add_argument('-r', type=float, default=0.0, help='Fixed r value (mm)')
parser.add_argument('-t', '--theta', type=float, default=0.0, help='Fixed theta value (deg)')
parser.add_argument('-z', type=float, default=0.0, help='Fixed z value (mm)')
parser.add_argument('-o', '--output', type=str, required=True, help='Output CSV file to append results.')

args = parser.parse_args()
freq = args.freq * 1e3
coordinate = args.sweep
r_val = args.r
theta_val = args.theta
z_val = args.z
output_file = args.output

CH1_ATTEN = 10      # Power Supply Voltage
CH2_ATTEN = 1/50e-3 # Current Probe 50 mV/A
CH3_ATTEN = 500     # TX Coil Voltage
CH4_ATTEN = 10      # RX Coil Voltage

SCOPE_SAMPLE_RATE = 2e7 
SCOPE_BUFFER_SIZE = 2000 

FORCE_PIN = 2
FORCE_FREQ = 1e3
PIN_SAMPLE_RATE = 1e6
NUM_PERIODS = 2
PIN_BUFFER_SIZE = int(PIN_SAMPLE_RATE / FORCE_FREQ * NUM_PERIODS) 

current_coords = {
    'r': r_val,
    'theta': theta_val,
    'z': z_val
}

print("="*40)
print(f" EXPERIMENT: Sweeping '{coordinate}' axis")
print(f' Fixed Vars: r={r_val} mm, theta={theta_val} deg, z={z_val} mm')
print(f" Frequency: {freq/1000:.1f} kHz")
print(f" Saving to: {output_file}")
print("="*40)

input('Press Enter to start the experiment...')

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

    pattern = device.digital_output
    pattern[0].setup_clock(frequency=freq, configure=True, start=True)
    device.digital_io[1].setup(enabled=True, state=True, configure=True)

    logic = device.digital_input
    logic.setup_edge_trigger(channel=FORCE_PIN, edge='rising')

    try:
        while True:
            coord_val = input(f'\nEnter value for {coordinate} or (or "q"): ').strip()

            if coord_val.lower() == 'q':
                print('Exiting...')
                break

            try:
                current_coords[coordinate] = float(coord_val)
            except ValueError:
                print(f'Invalid number')
                continue

            print(f'--> Gathering data at {coordinate}={current_coords[coordinate]}...')

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
                'r (mm)': current_coords['r'],
                'theta (deg)': current_coords['theta'],
                'z (mm)': current_coords['z'],
                'Driving Frequency (Hz)': freq,
                **results
            }
            append_data(output_file, data)

            
    except KeyboardInterrupt:
        print('Force quit detected. Exiting...')
    
    finally:
        device.digital_io[0].output_state = False
        device.digital_io[1].output_state = False
        device.close()