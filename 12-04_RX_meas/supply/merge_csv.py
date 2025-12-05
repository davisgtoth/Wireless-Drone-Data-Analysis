import argparse
import glob
import pandas as pd

parser = argparse.ArgumentParser(description="Merge CSV files into a single CSV")
parser.add_argument('-i', '--input', nargs='+', required=True, help='List or glob pattern of input CSV files to merge.')
parser.add_argument('-o', '--output', required=True, help='Output file for the merged CSV.')
args = parser.parse_args()

in_files = []
for pattern in args.input:
    in_files.extend(glob.glob(pattern))
out_file = args.output

file_names = [] 
supply_volt = []
tx_current = []
tx_amplitude = []
tx_frequency = []
rx_frequency = []
rx_average = []
supply_max = []
supply_min = []
current_freq = []
rx_max = []
rx_min = []
rx_rms = []

for file in in_files:
    name = file.split('/')[-1].split('.')[0]
    file_names.append(name)

    df = pd.read_csv(file, comment='#')
    data = df.to_numpy()

    for i, my_list in enumerate([supply_volt, tx_current, tx_amplitude, tx_frequency, rx_frequency, 
                                  rx_average, supply_max, supply_min, current_freq, rx_max, rx_min, rx_rms]):
        entry = data[i][-1].strip()
        value = float(entry.split(' ')[0])
        units = entry.split(' ')[1]

        if units[0] == 'm':  # milli
            value /= 1000.0
        elif units[0] =='k':  # kilo
            value *= 1000.0

        my_list.append(value)

data_dict = {
    'File Name': file_names,
    'Power Supply Avg (V)': supply_volt,
    'Power Supply Max (V)': supply_max,
    'Power Supply Min (V)': supply_min,
    'TX Current (A)': tx_current,
    'TX Current Freq (Hz)': current_freq,
    'TX Amplitude (V)': tx_amplitude,
    'TX Frequency (Hz)': tx_frequency,
    'RX Average (V)': rx_average,
    'RX RMS (V)': rx_rms,
    'RX Max (V)': rx_max,
    'RX Min (V)': rx_min,
    'RX Frequency (Hz)': rx_frequency
}

out_df = pd.DataFrame(data_dict)
out_df.to_csv(out_file, index=False, header=True)

print(f"Merged CSV saved to {out_file}")