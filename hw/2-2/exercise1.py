import numpy as np
import scipy.optimize

def V(r, k=1, c=1):
    # lengths are in angstroms
   return c*np.exp(k*r)/r

def DOCA(r, p, Er, V):
    return r**2*(1 - V(r)/Er) - p**2

# head on collision
x0=1 #angstrom
p=0
Er=10 #eV
root = scipy.optimize.fsolve(DOCA, x0=x0, args=(p, Er, V), xtol=1e-14)
print(f'The distance of closest approach is {float(root):.2f} angstroms.')
