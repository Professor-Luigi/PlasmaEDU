import numpy as np

from Efield import Efield
from BorisBunemann import BB


## Initialize
q = 1.6e-19 #C
m = 9.11e-31

# Trap dimensions
R = 0.01 #m
dr = R/100
dz = 1e-4
wall_voltage = 1e3
gap_length = 0.01 #m
cap_length = 0.01 #m
ring_length = 0.02 #m

efield = Efield(R, dr, dz, wall_voltage, gap_length, cap_length, ring_length)

# Bfield fn
def Bfield(x, y, z):
    B0 = 0.5 #T
    Bx, By, Bz = 0, 0, B0
    return Bx, By, Bz

Bx, By, Bz = Bfield(0, 0, 0)
Bmag = np.sqrt(Bz*Bz + By*By + Bx*Bx)

# Cyclotron
omega_c = np.abs(q)*Bmag/m
Tc = 2*np.pi/omega_c

# Initial velocity
vx0 = 0
vy0 = 0 
vz0 = 0

# Magnitude of v perp
v0 = np.sqrt(vx0*vx0 + vy0*vy0)

# Larmor Radius
r_L = v0/omega_c

# Initial Position
x0, y0, z0 = 0, 0, efield.z[efield.first_ring_index] + ring_length/2 

# Initial State Vector
X0 = np.array([x0, y0, z0, vx0, vy0, vz0])

# Time grid
Nperiods = 2
Nsteps_per_period = 15
time = np.linspace(0, Tc*Nperiods, Nperiods*Nsteps_per_period)
dt = time[1] - time[0]

# Boris Bunemann
bb = BB(time, X0, q, m, efield.e_interp, Bfield)
X_nc = bb.X_nc

plt.plot(X_nc[:,0], X_nc[:,1])
plt.show()
if __name__ == '__main__':
    pass
