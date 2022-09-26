import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(np.linspace(0, 5, 100), np.linspace(0, 5, 100)*6.26)
ax.set_xlim(0,30)
ax.set_ylim(0, ax.get_ylim()[1])
ax.set_title('Loss cone')
ax.set_ylabel('v$_{||}$ [v$_{th}$]')
ax.set_xlabel('v$_{perp}$ [v$_{th}$]')
fig.savefig('loss cone.png')
plt.show()
