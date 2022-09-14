'''
Application of the Boris-Bunemann solver for the Newton-Lorentz equation.
'''
import numpy as np
import matplotlib.pyplot as plt

qe = 1.602e-19
me = 9.01e-31
PROP_CYCLE = plt.rcParams['axes.prop_cycle']
COLORS = PROP_CYCLE.by_key()['color']

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

def freq_correction(x, corrected=True):
    # If we want the freq correction, corrected must be true, else is uncorrected
    alpha = 1
    if corrected and x != 0: # if x is 0, then tanx/x goes to 1
        alpha = np.tan(x)/x
    return alpha

def bb(time, X0, params, corrected=False):
    # does the boris bunemann integration
    # FOR THE FREQUENCY CORRECTION, THE MAGNETIC FIELD IS ASSUMED TO BE ALONG Z
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
        alpha_x = freq_correction(qmdt2*Bx, corrected=corrected)
        alpha_y = freq_correction(qmdt2*By, corrected=corrected)
        alpha_z = freq_correction(qmdt2*Bz, corrected=corrected)

        # Step 1 Half accel (Efield) v-
        vx += qmdt2*Ex*alpha_x
        vy += qmdt2*Ey*alpha_y
        vz += qmdt2*Ez*alpha_z

        # Step 2 Bfield rotation v' and v+
        tx = qmdt2*Bx*alpha_x
        ty = qmdt2*By*alpha_y
        tz = qmdt2*Bz*alpha_z
        tmagsq = tx*tx + ty*ty + tz*tz
        sx = 2*tx/(1 + tmagsq)
        sy = 2*ty/(1 + tmagsq)
        sz = 2*tz/(1 + tmagsq)
        vpx = vx + (vy*tz - vz*ty) 
        vpy = vy + (vz*tx - vx*tz)
        vpz = vz + (vx*ty - vy*tx) 
        vx += vpy*sz - vpz*sy
        vy += vpz*sx - vpx*sz
        vz += vpx*sy - vpy*sx

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

def velocity_push_back(X0, params, corrected=False):
    time = np.zeros(2) # will get the for loop in bb to run once
    X = bb(time, X0, params, corrected=corrected)
    return np.transpose(X)[3:,1] # this will give [vx vy vz] for the n-1/2 timestep

def bb_with_vel_push_back(time, X0, params, corrected=False):
    # Wrapper fn to include velocity push back to the BB integration
    # Velocity push back for 1st time step
    vx0, vy0, vz0 = velocity_push_back(X0, -.5*params, corrected=corrected)

    # Initial state vector with push back
    X0 = np.array([X0[0], X0[1], X0[2], vx0, vy0, vz0])

    # BB integration
    X = bb(time, X0, params, corrected=corrected)

    return X

def analytic(time, larmor_radius):
    # time is an array of times
    return larmor_radius*np.cos(time), larmor_radius*np.sin(time) 

def plot_trajectory(xs, ys, labels, title, larmor_radius, linestyles=None, markers=None, colors=None, show=True):
    '''
    xs, ys and labels will be lists where the first entry of each corresponds to one dataset,
    the next entry corresponds to the next dataset and so on.
    '''
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel('x/$r_L$')
    ax.set_ylabel('y/$r_L$')
    if linestyles is None:
        linestyles = ['-' for i in range(len(xs))]
    if markers is None:
        markers = ['' for i in range(len(xs))]
    if colors is None:
        colors = COLORS[:len(xs)]

    for x, y, label, ls, marker, color in zip(xs, ys, labels, linestyles, markers, colors):
        ax.plot(x/larmor_radius, y/larmor_radius, label=label, ls=ls, marker=marker, color=color)
    ax.legend()
    plt.grid()
    if '\n' in title:
        title = title.replace('\n', ' ')
    fig.savefig(f'{title}.png')
    if show:
        plt.show()
    plt.close()

def plot_errors(time, x, y, title, larmor_radius, show=True):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_ylabel('Absolute Error [$r_L$]')
    ax.set_xlabel('Larmor Period')
    ax.plot(time, x/larmor_radius, label='x(t)')
    ax.plot(time, y/larmor_radius, label='y(t)')
    ax.legend()
    fig.savefig(f'{title}.png')
    if show:
        plt.show()
    plt.close()

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
    v0 = np.sqrt(vx0*vx0 + vy0*vy0)

    # Larmor Radius
    r_L = v0/omega_c

    # Initial position [m]
    x0 = r_L
    y0 = 0
    z0 = 0

    # Initial state vector
    X0 = np.array([x0, y0, z0, vx0, vy0, vz0])

    # Time grid [s]
    Nperiods = 2
    Nsteps_per_period = 15
    time = np.linspace(0, Tc*Nperiods, Nperiods*Nsteps_per_period)
    dt = time[1] - time[0]

    # Parameters for bb
    params = np.array([dt, Q*dt/(M*2)])

    # BB integration with velocity push back
    X_uncorrected = bb_with_vel_push_back(time, X0, params, corrected=False)
    X_corrected   = bb_with_vel_push_back(time, X0, params, corrected=True)

    # Create the analytic solution
    analytic_time = np.linspace(0, 2*np.pi, 1000)
    analytic_x, analytic_y = analytic(analytic_time, r_L)

    # Calculate the absolute errors
    analytic_x_sampled, analytic_y_sampled = analytic(time, r_L)

    err_x_unc = np.abs(analytic_x_sampled - X_uncorrected[:,0])
    err_y_unc = np.abs(analytic_y_sampled - X_uncorrected[:,1])

    err_x_cor = np.abs(analytic_x_sampled - X_corrected[:,0])
    err_y_cor = np.abs(analytic_y_sampled - X_corrected[:,1])

    # Plot the analytic vs numerical solution (uncorr and corr in 2 subplots)
    plot_trajectory([analytic_x, X_uncorrected[:,0]],
                    [analytic_y, X_uncorrected[:, 1]],
                    ['Analytic', 'Boris-Bunemann w/o freq correction'],
                    'Analytic vs Boris-Bunemann without freq correction\nfor 1 Larmor Gyration',
                    r_L, 
                    markers=['', 'o'],
                    show=False)

    plot_trajectory([analytic_x, X_corrected[:,0]],
                    [analytic_y, X_corrected[:,1]],
                    ['Analytic', 'Boris-Bunemann with freq correction'],
                    'Analytic vs Boris Bunemann with freq correction\nfor 1 Larmor Gyration',
                    r_L,
                    linestyles=['-', 'dashed'],
                    markers=['', '*'],
                    colors=[COLORS[0], COLORS[2]],
                    show=False)

    plot_trajectory([X_uncorrected[:,0], X_corrected[:,0]],
                    [X_uncorrected[:,1], X_corrected[:,1]],
                    ['Boris-Bunemann w/o freq correction', 'Boris-Bunemann with freq correction'],
                    'Boris-Bunemann without vs with freq correction\nfor 1 Larmor Gyration',
                    r_L,
                    linestyles=['-', 'dashed'],
                    markers=['o', '*'],
                    colors=[COLORS[1], COLORS[2]],
                    show=False)

    # Plot the absolute errors for the numerical solutions
    plot_errors(time/Tc,
            err_x_unc,
            err_y_unc,
            'Absolute Error without Frequency Correction',
            r_L,
            show=True)

    plot_errors(time/Tc,
            err_x_cor,
            err_y_cor,
            'Absolute Error with Frequency Correction',
            r_L,
            show=False)
if __name__ == '__main__':
    main()
