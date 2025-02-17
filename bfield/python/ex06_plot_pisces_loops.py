################################################################################
#
#  BFIELD
#
#   Simple example of plot of the magnitude of the magnetic field
#   produced by an old published configuration of the PISCES experiment
#
#
################################################################################

import numpy as np
import bfield
import matplotlib.pyplot as plt

# Loops ( Ra,I0,Nturns, Xcenter,Ycenter,Zcenter, EulerAngles1,2,3 )
Loops = np.array([[ 0.25,2.5e4,1,  0.05,0,0, 90,0,0 ],
                  [ 0.25,-1.0e3,1, 0.30,0,0, 90,0,0 ],
                  [ 0.45,1.0e4,1,  0.65,0,0, 90,0,0 ],
                  [ 0.45,1.0e4,1,  1.15,0,0, 90,0,0 ],
                  [ 0.45,1.2e4,1,  1.65,0,0, 90,0,0 ] ])
Nloops = np.size(Loops,0)

X = np.linspace(  0.0, 1.7, 100 )
Y = np.linspace(  1e-10, 0.5, 100 )
Bnorm = np.zeros((X.size,Y.size))

for i in range(0,X.size):
  for j in range(0,Y.size):
    for k in range(0,Nloops):
      Ra     = Loops[k][0]
      I0     = Loops[k][1]
      Nturns = Loops[k][2]
      Center = Loops[k][3:6]
      Angles = Loops[k][6:9] * np.pi/180.0
      Point  = np.array([ X[i], Y[j], 0.0 ])
      Bx,By,Bz = bfield.loopxyz( Ra,I0,Nturns,Center,Angles,Point )
      Bnorm[i][j] += np.sqrt( Bx*Bx + By*By + Bz*Bz )

plt.figure(1)
plt.plot(X,Bnorm[:,1]*1e4)
plt.ylim(150,750)
plt.xlabel('Axis [m]')
plt.ylabel('B [Gauss]')
plt.title('B-field magnitude [T] - pisces')
plt.savefig('ex06_plot_pisces_loops.png',dpi=150)
plt.show()
