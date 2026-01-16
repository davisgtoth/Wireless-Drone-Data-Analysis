import argparse
import time
import numpy as np
import dwfpy as dwf
from tqdm import tqdm
from util.measurements import get_measurements
from util.append_data import append_data

parser = argparse.ArgumentParser(description="Perform a frequency sweep and log measurements.")
parser.add_argument('-c', '--coord', type=str, required=True, help='Coordinate varied in the measurement.')
parser.add_argument('-s', '--start', type=float, required=True, help='Start frequency in kHz.')
parser.add_argument('-e', '--stop', type=float, required=True, help='End frequency in kHz.')
parser.add_argument('-d', '--step', type=int, required=False, help='Step size in kHz.')
parser.add_argument('-o', '--output', type=str, required=True, help='Output CSV file to append results.')

args = parser.parse_args()
coordinate = args.coord
start_freq = args.start * 1e3
end_freq = args.stop * 1e3
step_size = args.step * 1e3 if args.step else 0.1e3 # default 0.1 kHz step
output_file = args.output

CH1_ATTEN = 10      # Power Supply Voltage
CH2_ATTEN = 1/50e-3 # Current Probe 50 mV/A
CH3_ATTEN = 500     # TX Coil Voltage
CH4_ATTEN = 10      # RX Coil Voltage

SAMPLE_RATE = 2e7 
BUFFER_SIZE = 1000 

freq_list = np.arange(start_freq, end_freq + step_size, step_size)
num_steps = len(freq_list) 

print("="*40)
print(f" EXPERIMENT SETUP: {coordinate} SWEEP")
print(f" Frequency: {start_freq/1000:.1f} kHz -> {end_freq/1000:.1f} kHz")
print(f" Steps:     {num_steps} points ({step_size/1000:.2f} kHz step)")
print(f" Saving to: {output_file}")
print("="*40)

with dwf.Device() as device:
    print(f"--> Connected to: {device.name} {device.serial_number}\n")
    
    scope = device.analog_input
    scope[0].setup(range=5.0, offset=0.0) 
    scope[1].setup(range=5.0, offset=0.0) 
    scope[2].setup(range=5.0, offset=0.0) 
    scope[3].setup(range=5.0, offset=0.0) 

    scope.setup_edge_trigger(channel=0, mode='auto')
    scope.setup_edge_trigger(channel=1, mode='auto')
    scope.setup_edge_trigger(channel=2, mode='auto')
    scope.setup_edge_trigger(channel=3, mode='auto')

    pattern = device.digital_output
    device.digital_io[1].setup(enabled=True, state=True, configure=True)

    try:
        while True:
            coord_val = input(f'\nEnter {coordinate} (mm) or [or "q"]: ').strip()

            if coord_val.lower() == 'q':
                print('Exiting...')
                break

            print(f'--> Starting frequency sweep for {coordinate} = {coord_val} mm...')

            pbar = tqdm(freq_list, desc='Sweeping', unit='Hz', ncols=100)

            for freq in pbar:
                pattern[0].setup_clock(frequency=freq, configure=True, start=True)
                time.sleep(0.05)

                scope.single(sample_rate=SAMPLE_RATE, buffer_size=BUFFER_SIZE, configure=True, start=True)
                ch1 = scope[0].get_data() * CH1_ATTEN
                ch2 = scope[1].get_data() * CH2_ATTEN
                ch3 = scope[2].get_data() * CH3_ATTEN
                ch4 = scope[3].get_data() * CH4_ATTEN
                results = get_measurements(ch1, ch2, ch3, ch4, SAMPLE_RATE)

                data = {
                    f'{coordinate} Coordinate (mm)': coord_val,
                    'Driving Frequency (Hz)': freq
                }
                data.update(results)
                append_data(output_file, data)

                tx_rms = results['TX Voltage RMS (V)']
                pbar.set_postfix_str(f'Freq={freq/1e3:.1f}k | TX_RMS={tx_rms:.2f}V')


            print(f'--> Sweep complete for {coordinate} = {coord_val} mm.\n')

            device.digital_io[0].output_state = False

    except KeyboardInterrupt:
        print('Force quit detected. Exiting...')

    finally:
        device.digital_io[0].output_state = False
        device.digital_io[1].output_state = False
        device.close()