'''
Application of the Boris-Bunemann solver for the Newton-Lorentz equation.
Class implementation
'''
import numpy as np
import os
from tqdm import tqdm

class BB:
    '''
    Does boris-bunemann integration of the lorentz equation. Includes velocity pushback and frequency correction.
    '''
    def __init__(self, time, X0, q, m, E_fn, B_fn, deltas=None, bounds=None, coords='cartesian', path=None, save_X_data=True):
        '''
        time is an array of times

        X0 are the initial parameters in an array [x y z vx vy vz] WITHOUT velocity pushback

        q is the particle charge (can be signed)

        m is the particle mass

        E_fn is a function for the electric field that depends on (x, y, z)

        B_fn is the same as E_fn but for the magnetic field

        deltas is an array of grid spacings for the coordinates in the order of bounds. This must be defined if bounds is.
        dtheta can be given as 0.

        If bounds is a list of bounds [(xmin, xmax),(ymin, ymax),(zmin, zmax)], then the solver will check if the particle goes outside of the given bounds. These can also be cylindrical in r, theta, z. theta must go from -2pi to 2pi based on how the angle is found.

        coords gives the coordinate system of bounds 

        path | The path of the file running the BB integrator.
        '''
        self.bounds = bounds
        self.coords = coords
        self.deltas = deltas
        if path is not None:
            self.path = path
        else:
            self.path = __file__
        # Make the params part of the instance
        self.dt = time[1] - time[0]
        self.n_time_steps = np.size(time)
        self.qmdt2 = q*self.dt/(m*2)
        params = np.array([self.dt, self.qmdt2])

        # Make the field functions part of the instance
        self.E_fn = E_fn
        self.B_fn = B_fn

        # Check to see if the X's are saved already
        filename = path.split('\\')[-1].split('.')[0]
        if f'X corrected-srcFile_{filename}.npy' in os.listdir() and save_X_data:
            self.loadX()
        else:
            # Get the corrected initial velocities without frequency correction
            print('Solving velocity push backs...')
            v0_pushed_back_nc = self.velocity_push_back(X0, -.5*params, corrected=False)
            X0_pushed_back_nc = np.array([X0[0],                X0[1],                X0[2],
                                          v0_pushed_back_nc[0], v0_pushed_back_nc[1], v0_pushed_back_nc[2]])

            # Get the corrected initial velocities with frequency correction
            v0_pushed_back_c = self.velocity_push_back(X0, -.5*params, corrected=True)
            X0_pushed_back_c = np.array([X0[0],               X0[1],               X0[2],
                                         v0_pushed_back_c[0], v0_pushed_back_c[1], v0_pushed_back_c[2]])

            # Do the BB integration
            print('Solving with the uncorrected Boris-Bunemann...')
            self.X_nc = self.bb(X0_pushed_back_nc, params, corrected=False)
            print('Solving with the corrected Boris-Bunemann...')
            self.X_c = self.bb(X0_pushed_back_c, params, corrected=True)

            # Save the X_c and X_nc
            if save_X_data:
                self.saveX()

    def bb(self, X0, params, corrected=False, checkBounds=True, vpb_n_time_steps=0):
       # does the boris bunemann integration
        # FOR THE FREQUENCY CORRECTION, THE MAGNETIC FIELD IS ASSUMED TO BE ALONG Z
        dt, qmdt2 = params
        M = np.size(X0)
        X = np.zeros((self.n_time_steps,M))
        X[0,:] = X0
        x = X[0,0] 
        y = X[0, 1]
        z = X[0,2]
        vx = X[0,3] 
        vy = X[0,4]
        vz = X[0,5] 
        if vpb_n_time_steps == 0:
            n_time_steps = self.n_time_steps
        else:
            n_time_steps = vpb_n_time_steps
        for n in tqdm(range(n_time_steps-1), desc='BB Time Step'):
            Ex, Ey, Ez = self.E_fn(x,y,z)
            Bx, By, Bz = self.B_fn(x,y,z)
            alpha_x = self.freq_correction(qmdt2*Bx, corrected=corrected)
            alpha_y = self.freq_correction(qmdt2*By, corrected=corrected)
            alpha_z = self.freq_correction(qmdt2*Bz, corrected=corrected)

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
            
            # Check that the particle is in bounds if asked for
            if self.bounds is not None and checkBounds:
                x1, x2, x3 = x, y, z
                coords_array = ['x','y','z']
                if self.coords == 'cylindrical':
                    x1, x2, x3 = self.convert2cyl(x,y,z)
                    coords_array = ['r', 'theta', 'z']
                for i, pos in enumerate([x1,x2,x3]):
                    correction_factor = 1
                    if self.coords == 'cylindrical' and i == 0:
                        correction_factor = 0 # for the case rlim (0, R) the lower bound should stay 0

                    # the deltas add a layer of security
                    if pos < self.bounds[i][0] + self.deltas[i]*correction_factor or pos > self.bounds[i][1] - self.deltas[i]:
                        # Freeze the particle at the last point before it leaves the domain
                        print(f'Particle will escape in the {coords_array[i]} direction, with a value of {pos}.')  
                        X[n+1:, 0] = X[n,0]
                        X[n+1:, 1] = X[n,1]
                        X[n+1:, 2] = X[n,2]
                        X[n+1:, 3] = X[n,3]
                        X[n+1:, 4] = X[n,4]
                        X[n+1:, 5] = X[n,5]
                        return X
            # Store coords into X
            X[n+1, 0] = x
            X[n+1, 1] = y
            X[n+1, 2] = z
            X[n+1, 3] = vx
            X[n+1, 4] = vy
            X[n+1, 5] = vz
        return X

    def freq_correction(self, x, corrected=True):
        # If we want the freq correction, corrected must be true, else is uncorrected
        alpha = 1
        if corrected and x != 0: # if x is 0, then tanx/x goes to 1
            alpha = np.tan(x)/x
        return alpha

    def velocity_push_back(self, X0, params, corrected=False):
        X = self.bb(X0, params, corrected=corrected, checkBounds=False, vpb_n_time_steps=2) # time steps will get the for loop in bb to run once
        return np.transpose(X)[3:,1] # this will give [vx vy vz] for the first n-1/2 timestep

    def convert2cyl(self, x, y, z):
        # x, y, z are floats
        r = np.linalg.norm(np.array([x,y]))
        if x == 0 and y != 0:
            theta = np.pi/2
        elif x == 0 and y == 0:
            theta = 0
        else:
            theta = np.arctan(y/x)
        return r, theta, z    

    def saveX(self):
        filename = self.path.split('\\')[-1].split('.')[0]
        with open(f'X corrected-srcFile_{filename}.npy', 'wb') as f:
            np.save(f, self.X_c)
        with open(f'X uncorrected-srcFile_{filename}.npy', 'wb') as f:
            np.save(f, self.X_nc)

    def loadX(self):
        filename = self.path.split('\\')[-1].split('.')[0]
        with open(f'X corrected-srcFile_{filename}.npy', 'rb') as f:
            self.X_c = np.load(f)
        with open(f'X uncorrected-srcFile_{filename}.npy', 'rb') as f:
            self.X_nc = np.load(f) 

