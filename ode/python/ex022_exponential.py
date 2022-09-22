import ode
import numpy as np
import matplotlib.pyplot as plt

def fun(x,y):
    ydot = np.exp(x)
    return ydot

def main():
    xn = np.linspace(0, 2, 5)
    y0 = np.array([1, 0])
    y_ef = ode.euler(fun, xn, y0)[:,0]
    y_mp = ode.midpoint(fun, xn, y0)[:,0]
    y_rk = ode.rk4(fun, xn, y0)[:,0]
    y_an = np.exp(xn)

    #errors
    err_ef = np.abs(y_an - y_ef)
    err_mp = np.abs(y_an - y_mp)
    err_rk = np.abs(y_an - y_rk)

    fig, ax = plt.subplots(2)
    plt.subplots_adjust(hspace=.6)
    ax[0].set_title('Approximations of Exp')
    ax[0].plot(xn, y_ef, 'ro-', label='Forward Euler (1st)')
    ax[0].plot(xn, y_mp, 'go-', label='Explicit Mid-Point (2nd)')
    ax[0].plot(xn, y_rk ,'bx-', label='Runge-Kutta (4th)')
    ax[0].plot(xn, y_an, 'k-', label='Analytical Solution')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].legend()

    ax[1].plot(xn, err_ef, 'ro-')
    ax[1].plot(xn, err_mp, 'go-')
    ax[1].plot(xn, err_rk, 'bx-')
    ax[1].set_title('Absolute Errors')
    ax[1].set_yscale('log')
    fig.savefig('ex022_exponential.png')
    plt.show()

if __name__ == '__main__':
    main()
