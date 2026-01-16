# import time
import dwfpy as dwf
import matplotlib.pyplot as plt

plt.ion()

with dwf.Device() as device:
    # print(f"Found device: {device.name} ({device.serial_number})")

    # print("Generating a 1kHz sine wave on WaveGen channel 1...")
    device.analog_output[0].setup('square', frequency=1e3, amplitude=1, start=True)

    # print("Preparing to read Ch2...")
    scope = device.analog_input
    scope[0].setup(range=10.0, offset=0.0)
    scope[1].setup(range=5.0, offset=0.0)
    # scope.configure()
    scope.setup_edge_trigger(channel=1, mode='auto')
    scope.single(sample_rate=1e6, buffer_size=4096, configure=True, start=True)
    ch2 = scope[1].get_data()
    ch1 = scope[0].get_data()*10  # 10x attenuation

    # print("Press Ctrl+C to stop")

    # while True:
    #     time.sleep(1)
    #     scope.read_status()
        
        # print(f'Ch1: {scope[0].get_sample()*10:.3f} V') # 10x attentuation 
        # print(f'Ch2: {scope[1].get_sample()} V')

    t = [i / 1e6 for i in range(len(ch1))]
    plt.figure(figsize=(10, 6))
    plt.plot(t, ch1, label='Channel 1')
    plt.plot(t, ch2, label='Channel 2')
    plt.title(f"Scope Capture")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.grid(True)
    plt.legend()
    plt.show()
