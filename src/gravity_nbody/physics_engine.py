import numpy as np
from scipy.integrate import solve_ivp

def n_body_acceleration(state, N, G, masses, softening):
    """
    Computes the acceleration for N bodies.
    state: Flattened array [x1, y1, z1, ..., vx1, vy1, vz1, ...]
    N: Number of bodies
    G: Gravitational constant
    masses: Array of masses
    softening: Softening parameter to prevent singularities
    """
    pos = state[:3*N].reshape(N, 3)
    vel = state[3*N:].reshape(N, 3)
    acc = np.zeros((N, 3))

    for i in range(N):
        for j in range(N):
            if i != j:
                r_vec = pos[j] - pos[i]
                r_mag = np.sqrt(np.sum(r_vec**2))
                
                # Softening to prevent division by zero and unrealistic forces at close encounters
                effective_r = np.sqrt(r_mag**2 + softening**2)
                
                f_mag = (G * masses[j]) / (effective_r**3)
                acc[i] += f_mag * r_vec
                
    return np.hstack((vel.flatten(), acc.flatten()))

class PhysicsEngine:
    def __init__(self, units='SI', method='DOP853', rtol=1e-9, atol=1e-9, softening=0.0):
        """
        Initialize the PhysicsEngine.
        
        units: 'SI' or 'AU_Year_SolarMass'
        method: Integration method for solve_ivp (default: 'DOP853')
        rtol: Relative tolerance for integration
        atol: Absolute tolerance for integration
        softening: Softening parameter for gravity (in distance units)
        """
        self.units = units
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.softening = softening
        
        # Constants
        if self.units == 'SI':
            self.G = 6.67430e-11  # m^3 kg^-1 s^-2
        elif self.units == 'AU_Year_SolarMass':
            # G in AU^3 * M_sun^-1 * yr^-2
            self.G = 4 * np.pi**2 # Approximately, for AU-Year-SolarMass
        else:
            raise ValueError(f"Unknown unit system: {self.units}")

    def get_gravitational_constant(self):
        return self.G

    def set_units(self, units):
        self.units = units
        if self.units == 'SI':
            self.G = 6.67430e-11
        elif self.units == 'AU_Year_SolarMass':
            self.G = 4 * np.pi**2 
        else:
            raise ValueError(f"Unknown unit system: {self.units}")

    def compute_derivatives(self, t, state, masses):
        """
        Wrapper for the N-body acceleration function to be compatible with solve_ivp.
        """
        N = len(masses)
        return n_body_acceleration(state, N, self.G, masses, self.softening)

    def run_simulation(self, initial_state, t_span, masses, t_eval=None):
        """
        Run the N-body simulation.
        
        initial_state: Flattened array [pos, vel]
        t_span: (t_start, t_end)
        masses: Array of masses
        t_eval: Array of time points to evaluate (optional)
        """
        
        # Ensure masses is a numpy array and correct type
        masses = np.array(masses, dtype=np.float64)
        
        sol = solve_ivp(
            fun=self.compute_derivatives,
            t_span=t_span,
            y0=initial_state,
            args=(masses,),
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
            t_eval=t_eval
        )
        
        return sol

    def calculate_kinetic_energy(self, masses, velocities):
        """
        Calculates the total kinetic energy of the system.
        T = 0.5 * sum(m_i * v_i^2)
        """
        # velocities shape: (N, 3) or flattened (3*N,)
        velocities = velocities.reshape(-1, 3)
        # v_squared = vx^2 + vy^2 + vz^2
        v_squared = np.sum(velocities**2, axis=1)
        return 0.5 * np.sum(masses * v_squared)

    def calculate_potential_energy(self, masses, positions):
        """
        Calculates the total potential energy of the system.
        U = - sum(G * m_i * m_j / r_ij) for i < j
        """
        positions = positions.reshape(-1, 3)
        N = len(masses)
        potential_energy = 0.0
        
        for i in range(N):
            for j in range(i + 1, N):
                r_vec = positions[i] - positions[j]
                r_mag = np.sqrt(np.sum(r_vec**2))
                # Add softening if needed, though usually strictly U doesn't have it unless specified
                # Using softening here to be consistent with dynamics if desired, 
                # but standard formula usually omits it for pure Newtonian gravity.
                # We will use the softening parameter from the class.
                effective_r = np.sqrt(r_mag**2 + self.softening**2)
                if effective_r > 0:
                    potential_energy -= (self.G * masses[i] * masses[j]) / effective_r
                    
        return potential_energy

    # Aliases for backward compatibility with existing notebooks
    _calculate_kinetic_energy = calculate_kinetic_energy
    _calculate_potential_energy = calculate_potential_energy
