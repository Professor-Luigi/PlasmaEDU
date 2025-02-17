import numpy as np
from matplotlib import patches
import matplotlib.pyplot as plt
import os
import scipy.stats as ss

from Efield import Efield
from BorisBunemann import BB

#------------------------------------------------------------------------------
# Functions
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

def didEscape(X, dt):
    escaped = False
    escape_params = []
    # Check if the entire state vector is the same for 2 subsequent time steps
    for i in range(1, len(X[:,0])-1):
        if np.all(X[i,:] == X[i+1,:]): 
            escaped = True
            escape_params.append(dt*i)
            escape_params.append(X[i,:3])
            break
    return escaped, escape_params 

def Maxwell(Nparticles, T, m):
    # Maxwell Dist returns a vx, vy, vz for Nparticles
    a = np.sqrt(1.38e-23*T/m)
    return ss.norm.rvs(size=Nparticles, scale=a), ss.norm.rvs(size=Nparticles, scale=a), ss.norm.rvs(size=Nparticles, scale=a)
#------------------------------------------------------------------------------
# Inputs

# Charge parameters for positrons/electron beam
q = -1.6e-19
m = 9.11e-31

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

# Set the initial positions of the particles rows are each particle, columns are x,y,z,vx,vy,vz
X0s = np.zeros((Nparticles, 6))
X0s[:,0], X0s[:,1], X0s[:,2] = np.ones(Nparticles)*0, np.ones(Nparticles)*0, np.ones(Nparticles)*dz*10 

# Thermal velocity
v_th = np.sqrt(2*1.38e-23*300/m)

# Beam characteristics
v_beam = np.sqrt(2*1.6e-19*(1000)/m) # velocity of beam electrons
beam_angles = np.linspace(0, np.pi/2, Nparticles)

# Set the initial velocities for the Nparticles
X0s[:,3], X0s[:,4], X0s[:,5] = np.zeros(Nparticles), v_beam*np.sin(beam_angles), v_beam*np.cos(beam_angles)#np.linspace(0, v_th*7, Nparticles)#Maxwell(Nparticles, 300, m) 

# Time grid inputs
Nperiods = 5000
Nsteps_per_period = 10
#------------------------------------------------------------------------------
# Fields

# Efield
efield = Efield(R, dr, dz, wall_voltage, gap_length, cap_length, ring_length, wall_thickness, voltage_error_convergence=5e-1)

# Bfield
def Bfield(x,y,z):
    B0 = 0.5 #T
    Bx, By, Bz = 0,0,B0
    return Bx, By, Bz
Bmag = np.linalg.norm(Bfield(0,0,0))
#------------------------------------------------------------------------------
# Particle motion constants

# Cyclotron
omega_c = np.abs(q)*Bmag/m
Tc = 2*np.pi/omega_c
#------------------------------------------------------------------------------
# Simulation

# Initial position array is X0s
# Rows are each particle, columns are x,y,z,vx,vy,vz

# Time grid
time = np.linspace(0, Tc*Nperiods, Nperiods*Nsteps_per_period)
dt = time[1] - time[0]

#Boris Bunemann
bb = [] # list of BB instances
X = [] # list of the X variables
escaped_bool = [] # whether the particle escaped or not
escaped_params = [] # time and point of escape
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
                         save_X_data=True,
                         load_X_data=True)
        bb.append(BB_instance)
        saveX(BB_instance.X_c, BB_instance.X_nc, __file__, i)
    else:
        X.append(loadX(__file__, i)[1]) # 1 is non frequency corrected X      
        single_escaped, single_escaped_params = didEscape(X[-1], dt) 
        escaped_bool.append(single_escaped)
        if single_escaped:
            escaped_params.append(single_escaped_params)
        else:
            escaped_params.append((None,None))
#------------------------------------------------------------------------------
# Plotting

linestyles = ['-', 'dashed', 'dotted', 'dashdot']
figsize=(14,7)

# xy of trap 
fig, ax = plt.subplots(figsize=figsize)
try:
    for i in range(Nparticles):
        ax.plot(bb[i].X_nc[:,0]*100, bb[i].X_nc[:,1]*100, label=i, ls=linestyles[int(i//10)])
except:
    for i in range(Nparticles):
        ax.plot(X[i][:,0]*100, X[i][:,1]*100, label=i, ls=linestyles[int(i//10)])
ax.plot(R*100*np.cos(np.linspace(0,2*np.pi,100)), R*100*np.sin(np.linspace(0,2*np.pi,100)), color='k')
ax.plot((R - wall_thickness)*100*np.cos(np.linspace(0,2*np.pi,100)), (R - wall_thickness)*100*np.sin(np.linspace(0,2*np.pi,100)), color='gray')
ax.set_xlabel('x (cm)')
ax.set_ylabel('y (cm)')
ax.set_title('XY plane trajectory of 20 positrons')
ax.legend(bbox_to_anchor=(1,1), loc='upper left')
fig.savefig('XY.png')

# z vs t of trap
fig, ax = plt.subplots(figsize=figsize)
try:
    for i in range(Nparticles):
        ax.plot(time, bb[i].X_nc[:,2]*100, label=f'{X0s[i,5]/v_th:.2f}'+'v$_{th}$', ls=linestyles[int(i//10)])
        #ax.plot(time, bb[i].X_nc[:,2]*100, label=i, ls=linestyles[int(i//10)])
except:
    for i in range(Nparticles):
        ax.plot(time, X[i][:,2]*100, label=f'{X0s[i,5]/v_th:.2f}'+'v$_{th}$', ls=linestyles[int(i//10)])
ax.axhline(0, color='k', lw=2)
ax.axhline((cap_length + gap_length)*100, color='gray', lw=2)
ax.axhline((cap_length + gap_length + ring_length)*100, color='gray', lw=2)
ax.axhline(trap_length*100, color='k', lw=2)
ax.set_xlabel('time (s)')
ax.set_ylabel('z (cm)')
ax.legend(bbox_to_anchor=(1,1), loc='upper left')
ax.set_title('z vs time trajectory of 20 positrons')
fig.savefig('zt.png')

# y vs z of beam in trap
fig, ax = plt.subplots(figsize=figsize)
try:
    for i in range(Nparticles):
        ax.plot(bb[i].X_nc[:,2]*100, bb[i].X_nc[:,1]*100, label=f'{beam_angles[i]*180/np.pi:.1f}'+'\u00B0', ls=linestyles[int(i//10)])
except:
    for i in range(Nparticles):
        ax.plot(X[i][:,2]*100, X[i][:,1]*100, label=f'{beam_angles[i]*180/np.pi:.1f}'+'\u00B0', ls=linestyles[int(i//10)])
ax.axvline(0, color='k', lw=2)
ax.axvline(trap_length*100, color='k', lw=2)
ax.axvline((cap_length + gap_length)*100, color='gray', lw=2)
ax.axvline((cap_length + gap_length + ring_length)*100, color='gray', lw=2)
ax.axhline(-R*100, color='k', lw=2)
ax.axhline(R*100, color='k', lw=2)
ax.set_xlabel('z (cm)')
ax.set_ylabel('y (cm)')
ax.legend(bbox_to_anchor=(1,1), loc='upper left')
ax.set_title('Incident E beam at various azimuthal angles')
fig.savefig('yz.png')

# y vs z of beam in trap zoomed in
fig, ax = plt.subplots(figsize=figsize)
try:
    for i in range(Nparticles):
        ax.plot(bb[i].X_nc[:,2]*100, bb[i].X_nc[:,1]*100, label=f'{beam_angles[i]*180/np.pi:.1f}'+'\u00B0', ls=linestyles[int(i//10)])
except:
    for i in range(Nparticles):
        ax.plot(X[i][:,2]*100, X[i][:,1]*100, label=f'{beam_angles[i]*180/np.pi:.1f}'+'\u00B0', ls=linestyles[int(i//10)])
ax.set_xlabel('z (cm)')
ax.set_ylabel('y (cm)')
ax.set_xlim(0.1, .107)
ax.legend(bbox_to_anchor=(1,1), loc='upper left')
ax.set_title('Zoomed in incident E beam at various angles')
fig.savefig('yz_zoomed.png')

# Plot the amplitude of the electron beam
fig, ax = plt.subplots(figsize=figsize)
beam_amplitudes=[]
try:
    for i in range(Nparticles):
        y_max = bb[i].X_nc[:,1].max()
        y_min = bb[i].X_nc[:,1].min()
        beam_amplitudes.append(y_max - y_min)
except:
    for i in range(Nparticles):
        y_max = X[i][:,1].max()
        y_min = X[i][:,1].min()
        beam_amplitudes.append(y_max - y_min)

print(beam_angles*180/np.pi, np.array(beam_amplitudes)*100)
ax.plot(beam_angles*180/np.pi, np.array(beam_amplitudes)*100)
ax.set_xlabel('Azimuthal angle ('+'\u00B0'+')')
ax.set_ylabel('Beam amplitude (cm)')
ax.set_title('Beam amplitude vs beam angle')
fig.savefig('beam amplitudes.png')
#plt.show()

# Plot the beam angles and energies on one plot
_1eV_amplitudes = [0.,
                  0.00011684,
                  0.00023288,
                  0.00034733,
                  0.0004594 ,
                  0.00056834,
                  0.0006734,
                  0.00077386,
                  0.00086903,
                  0.00095826,
                  0.00104095,
                  0.00111652,
                  0.00118447,
                  0.00124433,
                  0.00129569,
                  0.0013382,
                  0.00137157,
                  0.00139556,
                  0.00141003,
                  0.00141486
                  ]

_5eV_amplitudes = [0.,
                  0.00026126,
                  0.00052073,
                  0.00077665,
                  0.00102726,
                  0.00127085,
                  0.00150577,
                  0.00173039,
                  0.0019432,
                  0.00214273,
                  0.00232763,
                  0.00249662,
                  0.00264856,
                  0.00278241,
                  0.00289726,
                  0.00299231,
                  0.00306692,
                  0.00312058,
                  0.00315292,
                  0.00316373
                  ]

_10eV_amplitudes = [0.,
                   0.00036948,
                   0.00073643,
                   0.00109835,
                   0.00145276,
                   0.00179726,
                   0.00212948,
                   0.00244715,
                   0.0027481,
                   0.00303028,
                   0.00329176,
                   0.00353076,
                   0.00374564,
                   0.00393493,
                   0.00409734,
                   0.00423176,
                   0.00433727,
                   0.00441316,
                   0.0044589,
                   0.00447418
                   ]
 
_100eV_amplitudes = [0.,
                    0.00116838,
                    0.00232878,
                    0.00347328,
                    0.00459404,
                    0.00568343,
                    0.00673399,
                    0.00773855,
                    0.00869025,
                    0.00958259,
                    0.01040947,
                    0.01116524,
                    0.01184474,
                    0.01244333,
                    0.01295692,
                    0.013382,
                    0.01371566,
                    0.01395564,
                    0.01410028,
                    0.01414861
                    ]

_1000eV_amplitudes = [0.,
                     0.00369474,
                     0.00736425,
                     0.01098345,
                     0.01452762,
                     0.01797255,
                     0.02129471,
                     0.02447141,
                     0.02748094,
                     0.03030275,
                     0.0329176,
                     0.03530756,
                     0.03745633,
                     0.03934923,
                     0.04097334,
                     0.04231756,
                     0.04337274,
                     0.04413161,
                     0.04458901,
                     0.04474183
                     ]

fig, ax = plt.subplots(figsize=figsize)
ax.plot(beam_angles*180/np.pi, np.array(_1eV_amplitudes), label='1eV')
ax.plot(beam_angles*180/np.pi, np.array(_5eV_amplitudes), label='5eV')
ax.plot(beam_angles*180/np.pi, np.array(_10eV_amplitudes), label='10eV')
ax.plot(beam_angles*180/np.pi, np.array(_100eV_amplitudes), label='100eV')
ax.plot(beam_angles*180/np.pi, np.array(_1000eV_amplitudes), label='1000eV')
ax.legend()
ax.set_xlabel('Beam Angle (' + '\u00B0' + ')')
ax.set_ylabel('Beam Amplitude (m)')
ax.set_title('Beam Amplitude as a function of incident angle and energy')
ax.set_yscale('log')
fig.savefig('All Beam Amplitudes.png')

