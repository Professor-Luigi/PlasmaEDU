import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(1, '../../ode/python')
import ode

# Get the default color cycle for matplotlib
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


OMEGA = 1

# make the time array for the analytical solution
t = np.linspace(0, 2*np.pi/OMEGA*2, 100)

# Make the analytical solution
def analytical(t):
    return np.cos(OMEGA*t)

# Make the RK solution
tn = np.linspace(0, 2*np.pi/OMEGA*2, 20)
def fun(t, Y):
    y, vy = Y

    Ydot = np.zeros(2)
    Ydot[0] = vy
    Ydot[1] = -OMEGA**2*y
    return Ydot

y_rk = ode.rk4(fun, tn, np.array([0, 1])) 

# Make the Leapfrog without correction
dt = tn[1] - tn[0]
freq = 2/dt*np.arcsin(OMEGA*dt/2)#OMEGA

def lf(tn, freq):
    q = np.ones(tn.size)
    p = np.ones(tn.size) # shifted by half a sample
    dt = tn[1]-tn[0]

    q[0] = 1
    p[0] = 0

    # State transition matrix
    a = 1 - .5*freq**2*dt**2
    b = dt
    c = .25*freq**4*dt**3 - freq**2*dt
    phi = np.array([[a, b], [c, a]])

    for n in range(tn.size-1):
        q[n+1], p[n+1] = np.dot(phi, np.array([q[n], p[n]]))

    return q

y_lf_nc = lf(tn, freq) 

# Make the Leapfrog with correction
freq2 = OMEGA*np.sin(OMEGA*dt/2)/(OMEGA*dt/2)
y_lf_c = lf(tn, freq2)

# Make the plot of the solution
fig, ax = plt.subplots()
ax.set_title('Harmonic Oscillator Solutions')
ax.plot(tn, y_rk[:,1], marker='o', ls='dashed', label='Runge-Kutta 4th')
ax.plot(tn, y_lf_nc, marker='*', label='Leapfrog 2nd w/o freq correction')
ax.plot(tn, y_lf_c, marker='*', ls='dashed', label='Leapfrog 2nd w/ freq correction')
ax.plot(t, analytical(t), label='Analytical')
ax.set_ylabel('Position, x')
ax.set_xlabel('Time, t')
ax.legend()
fig.savefig('Solutions.png')
#plt.show()

# Make the plot of absolute error
err_rk = np.abs(y_rk[:,1] - analytical(tn))
err_lf_nc = np.abs(y_lf_nc - analytical(tn))
err_lf_c = np.abs(y_lf_c - analytical(tn))

fig, ax = plt.subplots()
ax.set_title('Absolute Error Comparison')
ax.set_xlabel('Time')
ax.set_ylabel('Absolute Error')
ax.plot(tn, err_rk, marker='o', ls='dashed', label='Runge-Kutta 4th')
ax.plot(tn, err_lf_nc, marker='*', label='Leapfrog 2nd w/o freq correction')
ax.plot(tn, err_lf_c, marker='*', ls = 'dashed', label='Leapfrog 2nd w/ freq correction')
ax.legend()
fig.savefig('All Abs Errs.png')
#plt.show()

fig, ax = plt.subplots()
ax.set_title('Absolute Error Comparison')
ax.set_xlabel('Time')
ax.set_ylabel('Absolute Error')
ax.plot(tn, err_rk, marker='o', ls='dashed', label='Runge-Kutta 4th')
ax.plot(tn, err_lf_c, marker='*', color=colors[2], ls = 'dashed', label='Leapfrog 2nd w/ freq correction')
ax.set_yscale('log')
ax.set_ylim(10**(-17), ax.get_ylim()[1])
ax.legend()
fig.savefig('Two Abs Errs.png')
#plt.show()

fig, ax = plt.subplots()
ax.set_title('Absolute Error')
ax.set_xlabel('Time')
ax.set_ylabel('Absolute Error')
ax.plot(tn, err_lf_c, marker='*', ls = 'dashed', color=colors[2], label='Leapfrog 2nd w/ freq correction')
ax.legend()
fig.savefig('One Abs Errs.png')
#plt.show()



