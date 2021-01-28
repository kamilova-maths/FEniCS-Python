import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0.0, 1.0, 0.01)
s = np.sin(2 * np.pi * t)
v = [1 + s, t]
fig = plt.figure()
plt.contour(v)
plt.ylabel('volts')
plt.xlabel('My x label')
plt.title('a sine wave')
# Fixing random state for reproducibility
# np.random.seed(19680801)
#
# ax2 = fig.add_axes([0.15, 0.1, 0.7, 0.3])
# n, bins, patches = ax2.hist(np.random.randn(1000), 50)
# ax2.set_xlabel('time (s)')

plt.show()