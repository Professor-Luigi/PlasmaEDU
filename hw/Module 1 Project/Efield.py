import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


class Efield:
    '''
    Solves for the electric field/voltage from the center of one end cap to the center of the ring.
    '''
    def __init__(self, R, dr, dz, v0, gap_length, cap_length, ring_length, **kwargs):
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

        self.r = np.arange(0, R, dr)
        self.z = np.arange(0, cap_length/2 + gap_length + ring_length/2, dz)
        self.Rpoints = np.size(self.r)
        self.Zpoints = np.size(self.z)

        self.voltage_error_convergence = kwargs['voltage_error_convergence']
        self.voltage_iterations = kwargs['voltage_iterations']

        self.v = self.v_solve(R, dr, dz, v0, gap_length, cap_length, error_convergence=1e-2, num_iterations=100)

    def v_solve(self, R, dr, dz, v0, gap_length, cap_length, error_convergence=1e-2, num_iterations=100):
        '''
        Solves for a cylindrically symmetric potential distribution. Specifically a penning trap.
        Origin is placed in the centroid of one of the endcaps
        All distances in meters.
        v0 is the potential of the endcaps.
        '''
        # Get important indicies
        self.last_cap_index = int((cap_length/2)//dz)
        self.first_gap_index = self.last_cap_index + 1
        self.last_gap_index = self.first_gap_index + int(gap_length//dz)
        self.first_ring_index = self.last_gap_index + 1

        # Initialize a voltage array
        v_last = np.ones((self.Zpoints, self.Rpoints))*0
        v_last[:self.last_cap_index+1,-1] = v0
        v_last[self.first_gap_index:self.first_ring_index+1, -1] = np.linspace(v0, 0, self.first_ring_index - self.first_gap_index + 1) 

        # Constants for the integration
        C = .5*dr*dr*dz*dz/(dr*dr + dz*dz)
        Cr = 1/(dr*dr)
        Cz = 1/(dz*dz)

        # Solve for the potential using sourceless Poisson eqn
        for iteration in range(num_iterations):
            v = np.copy(v_last)
            for m in range(1, self.Zpoints-1):
                for n in range(self.Rpoints-2, 0, -1):
                    v[m, n] = C*(  (v[m, n+1] - v[m, n-1])/(2*self.r[n]*dr)
                                 + (v[m, n+1] + v[m, n-1])*Cr
                                 + (v[m+1, n] + v[m-1, n])*Cz )

            # Set the r=0 BC
            v[:, 0] = v[:, 1]

            # Set the z=0 BC
            v[0,:] = v[1,:]

            error = np.abs(v - v_last)
            mean_error = np.mean(error)
            if mean_error < error_convergence:
                break
            v_last = np.copy(v)

        print(f'Voltage Iteration:{iteration+1}, Mean Error:{mean_error}')
        return v

R = 0.01 #m
dr = R/100 #m
dz = 1e-4 #m
v0 = 8

efield = Efield(R, dr, dz, v0, 0.01, 0.02, 0.03, voltage_error_convergence=5e-3, voltage_iterations=100)
v = efield.v
r = efield.r
z = efield.z
# Side view of the trap
fig, ax = plt.subplots()
mappable = ax.imshow(v, aspect='auto')
ax.axhline(efield.last_cap_index)
ax.axhline(efield.first_ring_index)
print(ax.get_yticks())
ax.set_ylabel('z (cm)')
ax.set_xlabel('Radius (cm)')
fig.colorbar(mappable)

# Plots for voltage
fig, ax = plt.subplots()
zstep=int(efield.Zpoints/4)
for v_rplane, z in zip(v[::zstep, :], z[::zstep]):
    ax.plot(r, v_rplane, label=f'z={z}') 
ax.set_title('V(r) for given z values')
ax.set_xlabel('Radius (m)')
ax.set_ylabel('Voltage (V)')
ax.legend()
plt.show()
