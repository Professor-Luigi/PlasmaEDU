import numpy as np
import matplotlib.pyplot as plt

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
wall_thickness = 0.005 #m
trap_length = 2*gap_length + 2*cap_length + ring_length #m

efield = Efield(R, dr, dz, wall_voltage, gap_length, cap_length, ring_length, wall_thickness, voltage_error_convergence=5e-1)

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
vy0 = 1e6 #m/s 
vz0 = 1e6 #m/s

# Magnitude of v perp
v0 = np.sqrt(vx0*vx0 + vy0*vy0)

# Larmor Radius
r_L = v0/omega_c

# Initial Position
x0, y0, z0 = 0, 0, efield.z[efield.first_ring_index] + ring_length/2 

# Initial State Vector
X0 = np.array([x0, y0, z0, vx0, vy0, vz0])

# Time grid
Nperiods = 500
Nsteps_per_period = 50
time = np.linspace(0, Tc*Nperiods, Nperiods*Nsteps_per_period)
dt = time[1] - time[0]

# Boris Bunemann
bb = BB(time, X0, q, m, efield.e_interp, Bfield, bounds=[(0,R), (-2*np.pi,2*np.pi), (0,trap_length)], coords='cylindrical', deltas=[dr, 0, dz])
X_nc = bb.X_nc
print(X_nc)

plt.figure()
plt.plot(X_nc[:,0], X_nc[:,1])
plt.plot(R*np.cos(np.linspace(0,2*np.pi,100)), R*np.sin(np.linspace(0, 2*np.pi,100)))
plt.xlabel('x (m)')
plt.ylabel('y (m)')

plt.figure()
plt.plot(time, X_nc[:,2])
plt.xlabel('time (s)')
plt.ylabel('z (m)')    
plt.show()
if __name__ == '__main__':
    pass
