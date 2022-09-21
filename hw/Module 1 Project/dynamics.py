import numpy as np

import Efield
import BorisBunemann


## Initialize
q = 1.6e-19 #C
m = 9.11e-31

B0 = 0.5 #T
Bx, By, Bz = 0, 0, B0
Bmag = np.sqrt(Bz*Bz + By*By + Bx*Bx)

# Cyclotron
omega_c = np.abs(q)*Bmag/m
Tc = 2*np.pi/omega_c

# Initial velocity
vx0 = 0
vy0 = 1e6
vz0 = 0

# Magnitude of v perp
v0 = np.sqrt(vx0*vx0 + vy0*vy0)


if __name__ == '__main__':
    pass
