import ode
import numpy as np
import matplotlib.pyplot as plt

def fun(x,y):
    ydot = y - x*x + 1.0
    return ydot

def main():
    xn   = np.linspace( 0.0, 5.0, 50 )     # Grid
    y0   = np.array( [ 0.5 ] )            # Initial condition
    y_ef = ode.euler( fun, xn, y0 )       # Forward Euler
    y_mp = ode.midpoint( fun, xn, y0 )    # Explicit Midpoint
    y_rk = ode.rk4( fun, xn, y0 )         # Runge-Kutta 4
    y_an = xn**2 + 2.0*xn + 1.0 - 0.5*np.exp(xn) # Analytical


    # Errors
    err_ef = np.abs(y_ef[:,0] - y_an)
    err_mp = np.abs(y_mp[:,0] - y_an)
    err_rk = np.abs(y_rk[:,0] - y_an)

    for i in range(0,xn.size):
        #print(xn[i], y_an[i], y_ef[i,0], y_mp[i,0], y_rk[i,0])
        pass

    plt.figure(1)
    plt.subplots_adjust(hspace=.6)
    plt.subplot(211)
    plt.plot( xn, y_ef, 'ro-', label='Forward Euler (1st)' )
    plt.plot( xn, y_mp, 'go-', label='Explicit Mid-Point (2nd)' )
    plt.plot( xn, y_rk, 'bx-', label='Runge-Kutta (4th)' )
    plt.plot( xn, y_an, 'k-',  label='Analytical Solution' )
    plt.title('Approximate solutions to ODE and Analytic')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc=3)

    plt.subplot(212)
    plt.plot(xn, err_ef, 'ro-')
    plt.plot(xn, err_mp, 'go-')
    plt.plot(xn, err_rk, 'bx-')
    plt.title('Absolute Errors')
    plt.yscale('log')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('ex02_ode_solution.png')
    plt.show()

if __name__ == '__main__':
   main()
