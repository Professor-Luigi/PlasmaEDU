################################################################################
#
#  BFIELD
#
#   Simple example of plot of the magnitude of the magnetic field
#   produced by multiple current loops
#
#
################################################################################

import numpy as np
import bfield
import matplotlib.pyplot as plt

# Loops ( Ra,I0,Nturns, Xcenter,Ycenter,Zcenter, EulerAngles1,2,3 )
Loops = np.array([[ 0.01,40,2,  0.04,0,0, 90,0,0 ],
                  [ 0.02,70,1,  0.03,0,0, 90,0,0 ],
                  [ 0.03,80,1,  0.02,0,0, 90,0,0 ],
                  [ 0.04,90,1,  0.01,0,0, 90,0,0 ],
                  [ 0.05,100,1,  0.00,0,0, 90,0,0 ],
                  [ 0.04,90,1, -0.01,0,0, 90,0,0 ],
                  [ 0.03,80,1, -0.02,0,0, 90,0,0 ],
                  [ 0.02,70,1, -0.03,0,0, 90,0,0 ],
                  [ 0.01,40,2, -0.04,0,0, 90,0,0 ] ])
Nloops = np.size(Loops,0)

X = np.linspace( -0.1, 0.1, 100 )
Y = np.linspace( -0.1, 0.1, 100 )
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
XX,YY = np.meshgrid(X,Y)
plt.contourf(np.transpose(XX),np.transpose(YY),Bnorm,30)
plt.colorbar()
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.title('B-field magnitude [T] - Multiple Loops')
plt.savefig('ex05_plot_multiple_loops.png',dpi=150)
#plt.show()
