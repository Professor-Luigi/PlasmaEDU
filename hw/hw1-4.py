'''
Application of the Boris-Bunemann solver for the Newton-Lorentz equation.
'''
import numpy as np
import matplotlib.pyplot as plt

qe = 1.602e-19
me = 9.01e-31

def Bfield(x, y, z):
    Bx = 0
    By = 0
    Bz = 0.1
    return Bx, By, Bz
def Efield(x, y, z):
    Ex = 0
    Ey = 0
    Ez = 0 
    return Ex, Ey, Ez

def freq_correction(x):
    alpha = np.tan(x)/x
    return alpha

def bb(time, X0, params):
    # does the boris bunemann integration
    dt, qmdt2 = params 
    N = np.size(time)
    M = np.size(X0)
    X = np.zeros((N,M))
    X[0,:] = X0
    x = X[0,0] 
    y = X[0, 1]
    z = X[0,2]
    vx = X[0,3] 
    vy = X[0,4]
    vz = X[0,5] 
    for n in range(N-1):
        Ex, Ey, Ez = Efield(x,y,z)
        Bx, By, Bz = Bfield(x,y,z)
        alpha_x = freq_correction(qmdt2*Bx)
        alpha_y = freq_correction(qmdt2*By)
        alpha_z = freq_correction(qmdt2*Bz)

        # Step 1 Half accel (Efield)
        vx += qmdt2*Ex*alpha_x
        vy += qmdt2*Ey*alpha_y
        vz += qmdt2*Ez*alpha_z

        # Step 2 Bfield rotation
        tx = qmdt2*Bx*alpha_x
        ty = qmdt2*By*alpha_y
        tz = qmdt2*Bz*alpha_z
        tmagsq = tx*tx + ty*ty + tz*tz
        sx = 2*tx/(1 + tmagsq)
        sy = 2*ty/(1 + tmagsq)
        sz = 2*tz/(1 + tmagsq)
        vpx = 
        vpy = 
        vpz = 
        vx +=
        vy +=
        vz +=

        # Step 3 Half accel (efield)
        vx += qmdt2*Ex*alpha_x
        vy += qmdt2*Ey*alpha_y
        vz += qmdt2*Ez*alpha_z

        # Step 4 Push Position
        x += vx*dt
        y += vy*dt
        z += vz*dt

        # Store coords into X
        X[n+1, 0] = x
        X[n+1, 1] = y
        X[n+1, 2] = z
        X[n+1, 3] = vx
        X[n+1, 4] = vy
        X[n+1, 5] = vz

    return X

def main():
    # Charge [C], mass [kg]
    Q = -qe
    M = me

    # Bfield [T]
    Bx, By, Bz = Bfield(0, 0, 0)
    Bmag = np.sqrt(Bx*Bx + By*By + Bz*Bz)

    # Cyclotron Freq [rad/s]
    omega_c = np.abs(Q)*Bmag/M

    # Cyclotron Period [s]
    Tc = 2*np.pi/omega_c

    # Initial velocity [m/s]
    vx0 = 0
    vy0 = 1e6
    vz0 = 0

    # Magnitude of v perp
    v0 = np.sqrt(vx*vx + vy*vy)

    # Larmor Radius
    r_L = v0/omega_c

    # Initial position [m]
    x0 = r_L
    y0 = 0
    z0 = 0

    # Initial state vector
    X0 = np.array([x0, y0, z0, vx0, vy0, vz0])

    # Time grid [s]
    Nperiods = 1
    Nsteps_per_period = 15
    time = np.linspace(0, Tc*Nperiods, Nperiods*Nsteps_per_period)
    dt = time[1] - time[0]

    # Parameters for bb
    params = np.array([dt, Q*dt/(M*2)])

    # Velocity push back for 1st time step
    vx0, vy0, vz0 = velocity_push_back(X0, -.5*params) # I implement this fn, can do BB but one step backwards

    # Initial state vector with push back
    X0 = np.array([x0, y0, z0, vx0, vy0, vz0])

    # BB integration
    X = bb(time, X0, params)






if __name__ == '__main__':
    main()
