import numpy as np
from matplotlib import patches
import matplotlib.pyplot as plt
import os
import scipy.stats as ss

from Efield import Efield
from BorisBunemann import BB

def saveX(X_c, X_nc, path, particle_num):
    filename = path.split('\\')[-1].split('.')[0]
    with open(f'multiparticleDynamics_X/X corrected-srcFile_{filename}-particle_{particle_num}.npy', 'wb') as f:
        np.save(f, X_c)
    with open(f'multiparticleDynamics_X/X uncorrected-srcFile_{filename}-particle_{particle_num}.npy', 'wb') as f:
        np.save(f, X_nc)

def loadX(path, particle_num):
    filename = path.split('\\')[-1].split('.')[0]
    with open(f'multiparticleDynamics_X/X corrected-srcFile_{filename}-particle_{particle_num}.npy', 'rb') as f:
        X_c = np.load(f)
    with open(f'multiparticleDynamics_X/X uncorrected-srcFile_{filename}-particle_{particle_num}.npy', 'rb') as f:
        X_nc = np.load(f)
    return X_c, X_nc 

def checkFiles(path, particle_num, override=False):
    # override == True makes the bb be rerun
    filename = path.split('\\')[-1].split('.')[0]
    File = f'X uncorrected-srcFile_{filename}-particle_{particle_num}.npy'
    shouldRecalculate = True
    if override:
        shouldRecalculate = True
    elif File in os.listdir('multiparticleDynamics_X'):
        shouldRecalculate = False 

    return shouldRecalculate

# Trap dimensions
R = 0.01
dr = R/100
dz = 1e-4
wall_voltage = 1e3
gap_length = 0.001
cap_length = 0.02
ring_length = 0.04
wall_thickness = 0.005
trap_length = 2*gap_length + 2*cap_length + ring_length

# Number of particles
Nparticles = 20

# Charge parameters for positrons
q = 1.6e-19
m = 9.11e-31

# Maxwell Dist returns a vx, vy, vz for Nparticles
def Maxwell(Nparticles, T, m):
    a = np.sqrt(1.38e-23*T/m)
    return ss.norm.rvs(size=Nparticles, scale=a), ss.norm.rvs(size=Nparticles, scale=a), ss.norm.rvs(size=Nparticles, scale=a)

# Set the initial positions of the particles rows are each particle, columns are x,y,z,vx,vy,vz
# Particles start somewhere in the ring up to R/2 m away from the center
X0s = np.zeros((Nparticles, 6))
X0s[:,0], X0s[:,1], X0s[:,2] = ss.uniform.rvs(size=Nparticles)*R - R/2, ss.uniform.rvs(size=Nparticles)*R- R/2, ss.uniform.rvs(size=Nparticles)*ring_length + cap_length + gap_length 

# Set the initial velocities for the Nparticles
X0s[:,3], X0s[:,4], X0s[:,5] = Maxwell(Nparticles, 300, m) 

# Efield
efield = Efield(R, dr, dz, wall_voltage, gap_length, cap_length, ring_length, wall_thickness, voltage_error_convergence=5e-1)

# Bfield
def Bfield(x,y,z):
    B0 = 0.5 #T
    Bx, By, Bz = 0,0,B0
    return Bx, By, Bz
Bmag = np.linalg.norm(Bfield(0,0,0))

# Cyclotron
omega_c = np.abs(q)*Bmag/m
Tc = 2*np.pi/omega_c

# Time grid
Nperiods = 500
Nsteps_per_period = 50
time = np.linspace(0, Tc*Nperiods, Nperiods*Nsteps_per_period)
dt = time[1] - time[0]

#Boris Bunemann
bb = [] # list of BB instances
X = [] # list of the X variables
for i in range(Nparticles):
    if checkFiles(__file__, i, override=False):
        print()
        print(f'Integrating for particle {i+1} of {Nparticles}')
        BB_instance = BB(time, X0s[i,:], q, m,
                         efield.e_interp, Bfield,
                         deltas=[dr, 0, dz],
                         bounds=[(0,R-wall_thickness),
                                 (-2*np.pi, 2*np.pi), (0, trap_length)],
                         coords='cylindrical',
                         path=__file__, 
                         save_X_data=False)
        bb.append(BB_instance)
        saveX(BB_instance.X_c, BB_instance.X_nc, __file__, i)
    else:
        X.append(loadX(__file__, i)[1]) # 1 is non frequency corrected X      
# xy of trap 
fig, ax = plt.subplots()
try:
    for i in range(Nparticles):
        ax.plot(bb[i].X_nc[:,0], bb[i].X_nc[:,1], label=i)
except:
    for i in range(Nparticles):
        ax.plot(X[i][:,0], X[i][:,1], label=i)
ax.plot(R*np.cos(np.linspace(0,2*np.pi,100)), R*np.sin(np.linspace(0,2*np.pi,100)), color='k')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('XY plane trajectory of 20 positrons')
ax.legend()
fig.savefig('XY.png')

# z vs t of trap
fig, ax = plt.subplots()
try:
    for i in range(Nparticles):
        ax.plot(time, bb[i].X_nc[:,2], label=i)
except:
    for i in range(Nparticles):
        ax.plot(time, X[i][:,2], label=i)
ax.axhline(0, color='k')
ax.axhline(trap_length, color='k')
ax.set_xlabel('time (s)')
ax.set_ylabel('z (m)')
ax.legend()
ax.set_title('z vs time trajectory of 20 positrons')
fig.savefig('zt.png')
plt.show()
