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
Np = 20
qs = np.concatenate((qe*np.ones(int(Np/2)) , -qe*np.ones(int(Np/2)))) # half protons half electrons
ms = np.concatenate((mp*np.ones(int(Np/2)),   me*np.ones(int(Np/2)))) 

# Initial positions and velocities (protons at rest electrons with random v [0,vbohr])
L = 100*a0
Rx = np.random.rand(Np)*L
Ry = np.random.rand(Np)*L
Rz = np.random.rand(Np)*L
Vx = np.concatenate((np.zeros(int(Np/2)), np.random.rand(int(Np/2))*vb))
Vy = np.concatenate((np.zeros(int(Np/2)), np.random.rand(int(Np/2))*vb))
Vz = np.concatenate((np.zeros(int(Np/2)), np.random.rand(int(Np/2))*vb))

# Nbody dynamics
def dynamics(time, y):
    Np = int(len(y)/6)
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
    print(ms, qs)
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
    T = 200*tb # sim time
    L = 100*a0 # characteristic size

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
    plt.plot(Rx[:, 0:int(Np/2)], Ry[:, 0:int(Np/2)], color='#03A9F4', marker='o', ls='-', label='Protons') #protons in blue 
    plt.plot(Rx[:, int(Np/2):Np], Ry[:, int(Np/2):Np], 'r-', label='Electrons') #electrons in red 
    plt.title('10 protons and 10 electrons\nN-body collision')
    #plt.legend(bbox_to_anchor=(1,1), loc='upper left')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.xlim(0, L)
    plt.ylim(0,L)
    plt.savefig('Nbody.png')
    plt.show()

if __name__ == '__main__':
    main(Rx, Ry, Rz, Vx, Vy, Vz)
