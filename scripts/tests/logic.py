import dwfpy as dwf
import numpy as np
import matplotlib.pyplot as plt

ESP32_FREQ = 5e3
PIN_SAMPLE_RATE = 1e6
NUM_PERIODS = 2
PIN_BUFFER_SIZE = int(PIN_SAMPLE_RATE / ESP32_FREQ * NUM_PERIODS)

with dwf.Device() as device:
    print(f"--> Connected to: {device.name} {device.serial_number}")

    device.analog_output[0].setup(
        'square', 
        frequency=ESP32_FREQ, 
        amplitude=1.65, 
        offset=1.65, 
        symmetry=75,
        start=True
    )

    pattern = device.digital_output
    device.digital_io[1].setup(enabled=True, state=True, configure=True)
    pattern[0].setup_clock(frequency=117*1e3, configure=True, start=True)

    input("Press Enter to continue...")

    # scope = device.analog_input
    # scope[0].setup(range=10.0, offset=0.0)
    # scope.setup_edge_trigger(channel=0, mode='auto')
    # scope.single(sample_rate=1e6, buffer_size=4096, configure=True, start=True)
    # ch1 = scope[0].get_data()

    logic = device.digital_input
    logic.setup_edge_trigger(channel=3, edge='rising')
    sample = logic.single(
        sample_rate=PIN_SAMPLE_RATE,
        buffer_size=PIN_BUFFER_SIZE,
        configure=True,
        start=True
    )
    logic.read_status()
    raw_data = logic.get_data()
    # pin0 = (raw_data >> 0) & 1
    # pin1 = (raw_data >> 1) & 1
    pin3 = (raw_data >> 3) & 1
    # print(len(pin_data))

    device.digital_io[0].output_state = False
    device.digital_io[1].output_state = False
    device.close()

duty_cycle = np.sum(pin3) / len(pin3) * 100
time_axis = np.linspace(0, PIN_BUFFER_SIZE/PIN_SAMPLE_RATE, len(pin3))
plt.figure(figsize=(10, 6))
# plt.step(time_axis * 1000, pin0, where='post', label='Pin 0')
# plt.step(time_axis * 1000, pin1, where='post', label='Pin 1')
plt.step(time_axis * 1000, pin3, where='post', label='Pin 3')
plt.title(f'Duty Cycle: {duty_cycle:.2f}%')
plt.grid(True)
plt.legend()
plt.show()

# t = [i / 1e6 for i in range(len(ch1))]
# plt.figure(figsize=(10, 6))
# plt.plot(t, ch1, label='Channel 1')
# plt.show()