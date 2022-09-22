import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pickle
import time


class Efield:
    '''
    Solves for the electric field/voltage from the edge of one end cap to the edge of the other.
    '''
    def __init__(self, R, dr, dz, v0, gap_length, cap_length, ring_length, wall_thickness, **kwargs):
        '''
        Accepted kwargs:
        voltage_error_convergence - largest average error between iterations
        voltage_iterations        - maximum number of iterations to solve the voltage "field"
        '''
        self.R = R
        self.dr = dr
        self.dz = dz
        self.v0 = v0

        self.gap_length = gap_length
        self.cap_length = cap_length
        self.ring_length = ring_length
        self.wall_thickness = wall_thickness

        self.r = np.arange(0, R, dr)
        self.z = np.arange(0, cap_length*2 + gap_length*2 + ring_length, dz)
        self.Rpoints = np.size(self.r)
        self.Zpoints = np.size(self.z)

        try:
            self.voltage_error_convergence = kwargs['voltage_error_convergence']
        except KeyError:
            self.voltage_error_convergence = 1e-1
        try:
            self.voltage_iterations = kwargs['voltage_iterations']
        except KeyError:
            self.voltage_iterations = 100

        self.v = self.v_solve(R, dr, dz, v0, gap_length, cap_length, ring_length, wall_thickness, error_convergence=self.voltage_error_convergence, num_iterations=self.voltage_iterations)
        self.Er, self.Et, self.Ez = self.e_solve()

    def v_solve(self, R, dr, dz, v0, gap_length, cap_length, ring_length, wall_thickness, error_convergence=1e-2, num_iterations=100):
        '''
        Solves for a cylindrically symmetric potential distribution. Specifically a penning trap.
        Origin is placed in the centroid of one of the endcaps
        All distances in meters.
        v0 is the potential of the endcaps.
        '''
        # Get important indicies
        self.first_cap_index = 0
        self.last_cap_index = int(cap_length//dz)

        self.first_gap_index = self.last_cap_index + 1
        self.last_gap_index = self.first_gap_index + int(gap_length//dz)

        self.first_ring_index = self.last_gap_index + 1
        self.last_ring_index = self.first_ring_index + int(ring_length//dz)

        self.first_gap_index2 = self.last_ring_index + 1
        self.last_gap_index2 = self.first_gap_index2 + int(gap_length//dz)

        self.first_cap_index2 = self.last_gap_index2 +1
        self.last_cap_index2 = self.Zpoints-1

        self.first_wall_index = int(self.Rpoints-wall_thickness/dr)
        # Initialize a voltage array
        v_last = np.ones((self.Zpoints, self.Rpoints))*0

        v_last[:self.last_cap_index+1,self.first_wall_index:] = v0 # cap voltage
        v_last[self.first_cap_index2:self.last_cap_index2+1,self.first_wall_index:] = v0 # cap voltage
        
        gap_n = self.first_ring_index - self.last_cap_index + 1
        wall_n = self.Rpoints - self.first_wall_index
        gap_n2 = self.first_cap_index2 - self.last_ring_index + 1
        v_last[self.last_cap_index:self.first_ring_index+1, self.first_wall_index:] = np.linspace(np.ones(wall_n)*v0,
                                                                                                  np.zeros(wall_n),
                                                                                                  gap_n) # in between 
        v_last[self.first_ring_index:self.last_ring_index+1, self.first_wall_index:] = 0 # Grounded ring
        v_last[self.last_ring_index:self.first_cap_index2+1, self.first_wall_index:] = np.linspace(np.zeros(wall_n),
                                                                                                   np.ones(wall_n)*v0,
                                                                                                   gap_n2) # in between

        # Constants for the integration
        C = .5*dr*dr*dz*dz/(dr*dr + dz*dz)
        Cr = 1/(dr*dr)
        Cz = 1/(dz*dz)

        # Solve for the potential using sourceless Poisson eqn
        total_time = 0
        for iteration in range(num_iterations):
            start_time = time.perf_counter()
            v = np.copy(v_last)
            for m in range(1, self.Zpoints-1):
                for n in range(self.Rpoints-2, 0, -1):
                    v[m, n] = C*(  (v[m, n+1] - v[m, n-1])/(2*self.r[n]*dr)
                                 + (v[m, n+1] + v[m, n-1])*Cr
                                 + (v[m+1, n] + v[m-1, n])*Cz )

            # Set the r=0 BC
            v[:, 0] = v[:, 1]

            # Set the z BCs
            v[0,:] = v[1,:]
            v[-1,:] = v[-2,:]

            # Reset the inner wall voltages (don't redo the linear decay in the gaps)
            v[:self.last_cap_index+1,self.first_wall_index:] = v0 # cap voltage
            v[self.first_cap_index2:self.last_cap_index2+1,self.first_wall_index:] = v0 # cap voltage
            v[self.first_ring_index:self.last_ring_index+1, self.first_wall_index:] = 0 # Grounded ring
            
            error = np.abs(v - v_last)
            mean_error = np.mean(error)
            print(f'Voltage Iteration:{iteration+1}, Mean Error:{mean_error:.3f}, Time:{time.perf_counter()-start_time:.3f}s')
            total_time += time.perf_counter() - start_time
            if mean_error < error_convergence:
                break
            v_last = np.copy(v)
        print(f'Total time:{total_time:.3f}s')
        return v

    def e_solve(self):
        Er = np.zeros((self.Zpoints, self.Rpoints))
        Et = np.zeros((self.Zpoints, self.Rpoints))
        Ez = np.zeros((self.Zpoints, self.Rpoints))
        # Solve for the electric field
        for m in range(self.Zpoints-2, 0, -1):
            for n in range(1, self.Rpoints-1):
                Er[m,n] = (self.v[m, n+1] - self.v[m, n-1])/(-2*self.dr)     
                Ez[m,n] = (self.v[m+1, n] - self.v[m-1, n])/(-2*self.dz)

        # Set the outermost points
        for E in (Er, Et, Ez):
            E[:, 0] = E[:, 1]
            E[:, -1] = E[:, -2]
            E[0, :] = E[1, :]
            E[-1, :] = E[-2, :]

        return Er, Et, Ez

    def e_interp(self, x1, x2, x3, coords='cartesian'):
        '''
        Interpolates the electric field at the inputted points
        coords=cylindrical: r, theta, z
        coords=cartesian: x, y, z
        '''
        r = x1
        if coords != 'cylindrical':
            r = np.sqrt(x1*x1 + x2*x2)
        # Obtain index along r, z coords
        n = np.abs(np.floor(r/self.dr)).astype(int)
        m = np.abs(np.floor(x3/self.dz)).astype(int)
        print(r, x3, n, m)

        # Get the coords of the closest node along r,z
        rn = n*self.dr
        zm = m*self.dz

        # Cell volumes
        A0 = (rn + self.dr - r)*(zm + self.dz - x3)
        A1 = (r - rn)*(zm + self.dz - x3)
        A2 = (r - rn)*(x3 - zm)
        A3 = (rn + self.dr - r)*(x3 - zm)
        Atot = self.dr*self.dz

        # Linear weights
        w0 = A0/Atot
        w1 = A1/Atot
        w2 = A2/Atot
        w3 = A3/Atot

        # Interpolate
        Er = w0*self.Er[m,n]     +\
             w1*self.Er[m,n+1]   +\
             w2*self.Er[m+1,n+1] +\
             w3*self.Er[m+1,n]  

        Ez = w0*self.Ez[m,n]     +\
             w1*self.Ez[m,n+1]   +\
             w2*self.Ez[m+1,n+1] +\
             w3*self.Ez[m+1,n]  
        
        # Finalize the interpolated Efield
        Ex1, Ex2, Ex3 = Er, 0, Ez
        if coords != 'cylindrical':
            Ex3 = Ez
            if x1 == 0 and x2 != 0:
                # Works for both pos and neg y axis
                theta = x2/np.abs(x2)*np.pi/2    
            elif x1 == 0 and x2 == 0:
                Er = 0 # true for the penning trap case
                theta = 0 # not technically true but avoids the 1/0 problem
            else:
                theta = np.arctan(x2/x1)

            Ex1, Ex2 = Er*np.cos(theta), Er*np.sin(theta) 

        return Ex1, Ex2, Ex3

if __name__ == '__main__':
    R = 0.01 #m
    dr = R/1000 #m
    dz = 1e-4 #m
    v0 = 1e3
    gap_length = 0.01
    cap_length = 0.02
    ring_length = 0.03
    wall_thickness = 0.001
    
    efield = Efield(R, dr, dz, v0, gap_length, cap_length, ring_length, wall_thickness, voltage_error_convergence=1e-1, voltage_iterations=100)
    v = efield.v
    r = efield.r
    z = efield.z
    Er, Et, Ez = efield.Er, efield.Et, efield.Ez

    rmesh, zmesh = np.meshgrid(np.arange(0, efield.Rpoints),
                               np.arange(0, efield.Zpoints))
    # Side view of the trap
    fig, ax = plt.subplots()
    mappable = ax.imshow(v, aspect='auto')
    ax.streamplot(rmesh, zmesh, Er, Ez, color='k', density=1, broken_streamlines=False)

    z_tick_spacing = 1e-2 #m
    z_ticks = np.arange(0, efield.Zpoints, z_tick_spacing//dz + 1).astype(int)
    z_ticklabels = z[z_ticks]/z_tick_spacing*10
    ax.set_yticks(z_ticks)
    ax.set_yticklabels(z_ticklabels.round(1))

    r_tick_spacing = 1e-3 #m
    r_ticks = np.arange(0, efield.Rpoints, r_tick_spacing//dr + 1).astype(int)
    r_ticklabels = r[r_ticks]/r_tick_spacing
    ax.set_xticks(r_ticks)
    ax.set_xticklabels(r_ticklabels.round(1))

    ax.axhline(efield.last_cap_index, linewidth=0.5)
    ax.axhline(efield.first_ring_index, linewidth=0.5)
    ax.axhline(efield.last_ring_index, linewidth=0.5)
    ax.axhline(efield.first_cap_index2, linewidth=0.5)
    ax.set_ylabel('z (mm)')
    ax.set_xlabel('Radius (mm)')
    fig.colorbar(mappable)
    fig.savefig('Side View.png')
    plt.show()

    # Plots for voltage
    fig, ax = plt.subplots()
    zstep=int(efield.Zpoints/4)
    for v_rplane, z in zip(v[::zstep, :], z[::zstep]):
        ax.plot(r, v_rplane, label=f'z={round(z,3)} m') 
    ax.set_title('V(r) for given z values')
    ax.set_xlabel('Radius (m)')
    ax.set_ylabel('Voltage (V)')
    ax.legend()
    fig.savefig('V vs r.png')
