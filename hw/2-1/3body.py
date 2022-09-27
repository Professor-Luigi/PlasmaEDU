import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(1, '../../ode/python/')
import ode

qe = 1.6e-19
me = 9.11e-31
mp = 1.6e-27
c = 299792458
hp = 6.63e-34
muref = 1e-7
mu0 = 4*np.pi*muref
eps0 = 1/c/c/mu0
fine = qe*qe*c*mu0/2/hp
kc = 1/4/np.pi/eps0
hbar = hp/2/np.pi

a0 = hbar/me/c/fine
mk = kc*qe*qe/me
vb = np.sqrt(mk/a0)
tb = 2*np.pi*np.sqrt(a0**3/mk)

# Particle properties
Np = 3 
qs = np.array([qe, -qe, -qe]) 
ms = np.array([mp, me, me]) 

Eiz = 13.6 #eV
# impact electron at energies: 1000, 30, Eiz, 7
E0_ev = 1000 
# Initial positions and velocities (protons at rest electrons with random v [0,vbohr])
Rx = np.array([0, a0, -10*a0])
Ry = np.array([0, 0, a0])
Rz = np.zeros(Np)
Vx = np.array([0, 0, np.sqrt(2*E0_ev*qe/me)])
Vy = np.array([0, vb, 0])
Vz = np.zeros(Np)

# Nbody dynamics
def dynamics(time, y):
    rx = y[0*Np:1*Np]
    ry = y[1*Np:2*Np]
    rz = y[2*Np:3*Np]
    vx = y[3*Np:4*Np]
    vy = y[4*Np:5*Np]
    vz = y[5*Np:6*Np]

    # clear a accumulators
    ax = np.zeros(Np)
    ay = np.zeros(Np)
    az = np.zeros(Np)
    # Accumulate forces
    for i in range(Np):
        for j in range(Np):
            if j!=i:
                rx_ij = rx[i] - rx[j]
                ry_ij = ry[i] - ry[j]
                rz_ij = rz[i] - rz[j]
                r_ij  = np.sqrt(rx_ij**2 + ry_ij**2 + rz_ij**2)
                Fx_ij = kc*qs[i]*qs[j]*rx_ij/(r_ij**3)
                Fy_ij = kc*qs[i]*qs[j]*ry_ij/(r_ij**3)
                Fz_ij = kc*qs[i]*qs[j]*rz_ij/(r_ij**3)
                ax[i] += Fx_ij/ms[i]
                ay[i] += Fy_ij/ms[i]
                az[i] += Fz_ij/ms[i]
    return np.concatenate((vx, vy, vz, ax, ay, az))

def main(Rx, Ry, Rz, Vx, Vy, Vz):
    T = 2*tb # sim time

    # time grid
    tspan = np.linspace(0, T, 200)

    # initial conditions
    Y0 = np.zeros((6*Np))
    Y0 = np.concatenate((Rx, Ry, Rz, Vx, Vy, Vz))
    
    # Solve ode
    Y = ode.rk4(dynamics, tspan, Y0)

    Rx = Y[:, 0*Np:1*Np]
    Ry = Y[:, 1*Np:2*Np]
    Rz = Y[:, 2*Np:3*Np]
    Vx = Y[:, 3*Np:4*Np]
    Vy = Y[:, 4*Np:5*Np]
    Vz = Y[:, 5*Np:6*Np]

    # plot
    plt.figure(figsize=(10,6))
    plt.plot(Rx[:, 0:int(Np/3)], Ry[:, 0:int(Np/3)], color='#03A9F4', marker='o', ls='-', label='Proton')
    plt.plot(Rx[:, int(Np/3):int(2*Np/3)], Ry[:, int(Np/3):int(2*Np/3)], 'r-', label='Bound Electron') 
    plt.plot(Rx[:, int(2*Np/3):Np], Ry[:, int(2*Np/3):Np], 'k-', label='Free Electron') 
    plt.title(f'Electron at energy {E0_ev}eV collision with H atom')
    plt.legend()
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.savefig(f'Electron at energy {E0_ev}eV collision with H atom.png')
    plt.show()

if __name__ == '__main__':
    main(Rx, Ry, Rz, Vx, Vy, Vz)
