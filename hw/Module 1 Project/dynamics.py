import numpy as np
import matplotlib.pyplot as plt
import tqdm

from Efield import Efield
from BorisBunemann import BB


#------------------------------------------------------------------------------
# Inputs

# Save/load Boris-Bunemann X data
save_bb = True
load_bb = False

# Particle characteristics
q = 1.6e-19 #C
m = 9.11e-31

# Trap dimensions
R = 0.01 #m
dr = R/100
dz = 1e-4
wall_voltage = 1e3
gap_length = 0.001 #m
cap_length = 0.02 #m
ring_length = 0.04 #m
wall_thickness = 0.005 #m
trap_length = 2*gap_length + 2*cap_length + ring_length #m

# Initial Position
x0, y0, z0 = -0.00275, -0.0037, 0.01 

# Thermal Velocity
v_th = np.sqrt(2*1.38e-23*300/m)

# Initial velocity
vx0 = 3.27e4 
vy0 = -1.02e5 #m/s 
vz0 = 5.5e4 #m/s

# Time grid inputs
Nperiods = 182
Nsteps_per_period = 50
#------------------------------------------------------------------------------
# Fields

# Efield fn
efield = Efield(R, dr, dz, wall_voltage, gap_length, cap_length, ring_length, wall_thickness, voltage_error_convergence=5e-1)

# Bfield fn
def Bfield(x, y, z):
    B0 = 0.5 #T
    Bx, By, Bz = 0, 0, B0
    return Bx, By, Bz

Bx, By, Bz = Bfield(0, 0, 0)
Bmag = np.sqrt(Bz*Bz + By*By + Bx*Bx)
#------------------------------------------------------------------------------
# Particle Motion constants

# Cyclotron
omega_c = np.abs(q)*Bmag/m
Tc = 2*np.pi/omega_c

# Magnitude of v perp
v0 = np.sqrt(vx0*vx0 + vy0*vy0)

# Larmor Radius
r_L = v0/omega_c

# ExB drift velocity
v_drift = np.linalg.norm(efield.e_interp(x0, y0, z0))/Bmag
#------------------------------------------------------------------------------
# Simulation

# Initial State Vector
X0 = np.array([x0, y0, z0, vx0, vy0, vz0])

# Time grid
time = np.linspace(0, Tc*Nperiods, Nperiods*Nsteps_per_period)
dt = time[1] - time[0]

# Boris Bunemann
bb = BB(time, X0, q, m, efield.e_interp, Bfield,
        bounds=[(0,R-wall_thickness), (-2*np.pi,2*np.pi), (0,trap_length)],
        coords='cylindrical',
        deltas=[dr, 0, dz],
        save_X_data=save_bb,
        load_X_data=load_bb,
        path=__file__)
X_nc = bb.X_nc
#------------------------------------------------------------------------------
# Plotting

# XY plot
fig, ax = plt.subplots()
ax.set_title('XY Plot of the single positron')
ax.plot(X_nc[:,0], X_nc[:,1])
ax.plot(R*np.cos(np.linspace(0,2*np.pi,100)), R*np.sin(np.linspace(0, 2*np.pi,100)), color='k')
ax.set_xlabel('x (m)')
ax.set_ylabel('y(m)')
fig.savefig('dynamics_xy.png')

# Zoomed in XY plot
fig, ax = plt.subplots()
ax.set_title('Zoomed in XY Plot of the single positron')
ax.plot(X_nc[:,0], X_nc[:,1])
ax.set_xlabel('x (m)')
ax.set_ylabel('y(m)')
fig.savefig('dynamics_xy_zoomed.png')

# Z vs t plot
fig, ax = plt.subplots()
ax.plot(time, X_nc[:,2])
ax.set_title('Time evolution of the z position')
ax.set_xlabel('time (s)')
ax.set_ylabel('z (m)')
ax.axhline(0, color='k')
ax.axhline(trap_length, color='k')
fig.savefig('dynamics_zt.png')
