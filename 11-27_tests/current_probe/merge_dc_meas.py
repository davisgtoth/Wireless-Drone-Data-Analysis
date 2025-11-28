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
probe = []
resistor = []
theory = []

for file in in_files:
    name = file.split('/')[-1].split('.')[0]
    file_names.append(name)

    df = pd.read_csv(file, comment='#')

    supply_val = df['Value'][0].strip()
    supply_val = float(supply_val.split(' ')[0])
    supply_volt.append(supply_val)

    probe_val = df['Value'][7].strip()
    probe_val = float(probe_val.split(' ')[0])
    probe.append(probe_val)

    res_val = df['Value'][8].strip()
    res_val = float(res_val.split(' ')[0])
    resistor.append(res_val)

    theory_val = df['Value'][9].strip()
    theory_val = float(theory_val.split(' ')[0])
    theory.append(theory_val)

data_dict = {
    'File Name': file_names,
    'Power Supply (V)': supply_volt,
    'Current Probe (A)': probe,
    'Resistor Meas. (A)': resistor,
    'Theoretical (A)': theory
}

out_df = pd.DataFrame(data_dict)
out_df.to_csv(out_file, index=False, header=True)

print(f"Merged CSV saved to {out_file}")
