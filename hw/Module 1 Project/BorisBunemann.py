'''
Application of the Boris-Bunemann solver for the Newton-Lorentz equation.
Class implementation
'''
import numpy as np

class BB:
    '''
    Does boris-bunemann integration of the lorentz equation. Includes velocity pushback and frequency correction.
    '''
    def __init__(self, time, X0, q, m, E_fn, B_fn):
        '''
        time is an array of times
        X0 are the initial parameters in an array [x y z vx vy vz] WITHOUT velocity pushback
        q is the particle charge (can be signed)
        m is the particle mass
        E_fn is a function for the electric field that depends on (x, y, z)
        B_fn is the same as E_fn but for the magnetic field
        '''
        # Make the params part of the instance
        self.dt = time[1] - time[0]
        self.n_time_steps = np.size(time)
        self.qmdt2 = q*self.dt/(m*2)
        params = np.array([self.dt, self.qmdt2])

        # Make the field functions part of the instance
        self.E_fn = E_fn
        self.B_fn = B_fn

        # Get the corrected initial velocities without frequency correction
        v0_pushed_back_nc = self.velocity_push_back(X0, -.5*params, corrected=False)
        X0_pushed_back_nc = np.array([X0[0],                X0[1],                X0[2],
                                      v0_pushed_back_nc[0], v0_pushed_back_nc[1], v0_pushed_back_nc[2]])

        # Get the corrected initial velocities with frequency correction
        v0_pushed_back_c = self.velocity_push_back(X0, -.5*params, corrected=True)
        X0_pushed_back_c = np.array([X0[0],               X0[1],               X0[2],
                                     v0_pushed_back_c[0], v0_pushed_back_c[1], v0_pushed_back_c[2]])

        # Do the BB integration
        self.X_nc = self.bb(X0_pushed_back_nc, params, corrected=False)
        self.X_c = self.bb(X0_pushed_back_c, params, corrected=True)


    def bb(self, X0, params, corrected=False):
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
        for n in range(self.n_time_steps-1):
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
        time = np.zeros(2) # will get the for loop in bb to run once
        X = self.bb(X0, params, corrected=corrected)
        return np.transpose(X)[3:,1] # this will give [vx vy vz] for the first n-1/2 timestep


