import dwfpy as dwf
import numpy as np
import matplotlib.pyplot as plt

ESP32_FREQ = 1e3
PIN_SAMPLE_RATE = 1e6
NUM_PERIODS = 2
PIN_BUFFER_SIZE = int(PIN_SAMPLE_RATE / ESP32_FREQ * NUM_PERIODS)
# print(f'Pin Buffer Size: {PIN_BUFFER_SIZE}')

with dwf.Device() as device:
    print(f"--> Connected to: {device.name} {device.serial_number}")

    device.analog_output[0].setup(
        'square', 
        frequency=ESP32_FREQ, 
        amplitude=1.65, 
        offset=1.65, 
        symmetry=13,
        start=True
    )

    pattern = device.digital_output
    device.digital_io[1].setup(enabled=True, state=True, configure=True)
    pattern[0].setup_clock(frequency=117*1e3, configure=True, start=True)

    input("Press Enter to continue...")

    # scope = device.analog_input
    # scope[3].setup(range=10.0, offset=0.0)
    # scope.setup_edge_trigger(channel=3, mode='auto')
    # scope.single(sample_rate=1e6, buffer_size=4096, configure=True, start=True)
    # ch4 = scope[3].get_data()

    logic = device.digital_input
    logic.setup_edge_trigger(channel=2, edge='rising')
    # logic.setup_edge_trigger(channel=2, edge='falling')
    sample = logic.single(
        sample_rate=PIN_SAMPLE_RATE,
        buffer_size=PIN_BUFFER_SIZE,
        # sample_rate=1e6,
        # buffer_size=4096,
        configure=True,
        start=True
    )
    # print(logic.read_status())
    # logic.read_status()
    raw_data = logic.get_data()
    print(raw_data)
    # pin0 = (raw_data >> 0) & 1
    # pin1 = (raw_data >> 1) & 1
    pin2 = (raw_data >> 2) & 1
    print(len(pin2))

    device.digital_io[0].output_state = False
    device.digital_io[1].output_state = False
    device.close()

duty_cycle = np.sum(pin2) / len(pin2) * 100
time_axis = np.linspace(0, PIN_BUFFER_SIZE/PIN_SAMPLE_RATE, len(pin2))
plt.figure(figsize=(10, 6))
# plt.step(time_axis * 1000, pin0, where='post', label='Pin 0')
# plt.step(time_axis * 1000, pin1, where='post', label='Pin 1')
plt.step(time_axis * 1000, pin2, where='post', label='Pin 2')
plt.title(f'Duty Cycle: {duty_cycle:.2f}%')
plt.grid(True)
# plt.legend()
plt.show()

# t = [i / 1e6 for i in range(len(ch4))]
# plt.figure(figsize=(10, 6))
# plt.plot(t, ch4, label='Channel 4')
# plt.show()