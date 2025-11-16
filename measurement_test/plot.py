import matplotlib.pyplot as plt

V = [5, 10, 15]
TX_I = [6.7537, 14.978, 21.064]
TX_V = [152.24, 286.18, 417.43]
RX_V = [257.7e-3, 0.50062, 0.72056]

plt.figure()
plt.plot(V, TX_I, 'o-', label='TX Current')
plt.xlabel('Power Supply Votlage (V)')
plt.legend()
plt.show()

plt.figure()
plt.plot(V, TX_V, 'o-', label='TX Voltage')
plt.xlabel('Power Supply Votlage (V)')
plt.legend()
plt.show()

plt.figure()
plt.plot(V, RX_V, 'o-', label='RX Voltage')
plt.xlabel('Power Supply Votlage (V)')
plt.legend()
plt.show()
