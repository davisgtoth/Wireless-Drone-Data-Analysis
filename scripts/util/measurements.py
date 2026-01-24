import numpy as np

def get_measurements(ch1, ch2, ch3, ch4, sample_rate, pin_data=None, force_pin=None):
    '''
    Returns a dictionary containing the measurements done on each channel. 
    Assumes all unit conversion/attenuation has already been applied to the channel data.
    '''
    meas = {}
    
    # Channel 1 - Power Supply Voltage (V) 
    meas.update({'Power Supply Average (V)': np.mean(ch1)})     # Average
    meas.update({'Power Supply Min (V)': np.min(ch1)})          # Min
    meas.update({'Power Supply Max (V)': np.max(ch1)})          # Max

    # Channel 2 - TX Current (A)
    meas.update({'TX Current RMS (A)': np.std(ch2)})                             # RMS
    meas.update({'TX Current Peak-to-Peak (A)': 2 * np.sqrt(2) * np.std(ch2)})   # Peak-to-Peak
    meas.update({'TX Current Average (A)': np.mean(ch2)})                        # Average
    meas.update({'TX Current Frequency (Hz)': get_freq(ch2, sample_rate)})       # Frequency

    # Channel 3 - TX Voltage (V)
    meas.update({'TX Voltage RMS (V)': np.std(ch3)})                             # RMS
    meas.update({'TX Voltage Peak-to-Peak (V)': 2 * np.sqrt(2) * np.std(ch3)})   # Peak-to-Peak
    meas.update({'TX Voltage Average (V)': np.mean(ch3)})                        # Average
    meas.update({'TX Voltage Frequency (Hz)': get_freq(ch3, sample_rate)})       # Frequency

    # Channel 4 - RX Voltage (V)
    meas.update({'RX Voltage Average (V)': np.mean(ch4)})           # Average
    meas.update({'RX Voltage RMS (V)': np.sqrt(np.mean(ch4)**2)})   # RMS
    meas.update({'RX Coil Min (V)': np.min(ch4)})                   # Min
    meas.update({'RX Coil Max (V)': np.max(ch4)})                   # Max
    
    # Digital I/O - Force Measurement 
    if pin_data is not None:
        signal = (pin_data >> force_pin) & 1
        duty_cycle = np.sum(signal) / len(signal) * 100
        meas.update({'RX Force (N)': duty_cycle*10 * 1e-3 * 9.81})  # duty cyle * 10 = mass in grams

    return meas


def get_freq(data, sample_rate):
    centred = data - np.mean(data)
    crossings = np.where(np.diff(np.sign(centred)) > 0)[0]

    if len(crossings) < 2:
        return float('nan')  # Not enough crossings to determine frequency

    points_per_cycle = (crossings[-1] - crossings[0]) / (len(crossings) - 1)
    freq = sample_rate / points_per_cycle

    return freq