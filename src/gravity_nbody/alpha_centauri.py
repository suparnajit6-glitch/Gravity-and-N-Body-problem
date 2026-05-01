import matplotlib.pyplot as plt
import numpy as np

from gravity_nbody.physics_engine import PhysicsEngine
from gravity_nbody.stardata import get_velocity_arrays as gva
from gravity_nbody.stardata import pcalc2 as pc2
from gravity_nbody.stardata import v_relative as vr


AU = 1.496e11
G = 6.67430e-11
N = 3


def main():
    year_seconds = 365.25 * 24 * 3600
    t_span = (0, 80 * year_seconds)
    t_points = 20000
    t_eval = np.linspace(t_span[0], t_span[1], t_points)

    m0 = 1.98847e30  # 1 solar-mass
    mass = np.array([1.1 * m0, 0.907 * m0, 0.122 * m0])

    # Initial Conditions
    pos00 = pc2()
    vel00 = gva()
    print("Initial Velocities (Absolute):", vel00)

    vel01 = vr()
    print("Initial Velocities (Relative):", vel01)

    state00 = np.hstack((pos00.flatten(), vel00.flatten()))
    state01 = np.hstack((pos00.flatten(), vel01.flatten()))

    # High accuracy requested: DOP853 with tight tolerances.
    engine = PhysicsEngine(units='SI', method='DOP853', rtol=1e-10, atol=1e-10)

    print("Running simulation for Original Solution...")
    sol = engine.run_simulation(state00, t_span, mass, t_eval=t_eval)

    print("Running simulation for Relative Solution...")
    sol0 = engine.run_simulation(state01, t_span, mass, t_eval=t_eval)

    fig = plt.figure(figsize=(14, 12))

    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(sol.y[0, :] / AU, sol.y[1, :] / AU, sol.y[2, :] / AU, 'orange', label='Alpha Cen A')
    ax1.plot(sol.y[3, :] / AU, sol.y[4, :] / AU, sol.y[5, :] / AU, 'blue', label='Alpha Cen B')
    ax1.set_xlabel('X (AU)')
    ax1.set_ylabel('Y (AU)')
    ax1.set_zlabel('Z (AU)')
    ax1.set_title('Alpha Cen A & B (original)')
    ax1.legend()

    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.plot(sol.y[0, :] / AU, sol.y[1, :] / AU, sol.y[2, :] / AU, 'orange', linewidth=0.5, label='Alpha Cen A')
    ax2.plot(sol.y[3, :] / AU, sol.y[4, :] / AU, sol.y[5, :] / AU, 'blue', linewidth=0.5, label='Alpha Cen B')
    ax2.plot(sol.y[6, :] / AU, sol.y[7, :] / AU, sol.y[8, :] / AU, 'red', linewidth=1.5, label='Proxima')
    ax2.set_xlabel('X (AU)')
    ax2.set_ylabel('Y (AU)')
    ax2.set_zlabel('Z (AU)')
    ax2.set_title('All three stars (original)')
    ax2.legend()

    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax3.plot(sol0.y[0, :] / AU, sol0.y[1, :] / AU, sol0.y[2, :] / AU, 'orange', label='Alpha Cen A')
    ax3.plot(sol0.y[3, :] / AU, sol0.y[4, :] / AU, sol0.y[5, :] / AU, 'blue', label='Alpha Cen B')
    ax3.set_xlabel('X (AU)')
    ax3.set_ylabel('Y (AU)')
    ax3.set_zlabel('Z (AU)')
    ax3.set_title('Alpha Cen A & B (relative)')
    ax3.legend()

    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    ax4.plot(sol0.y[0, :] / AU, sol0.y[1, :] / AU, sol0.y[2, :] / AU, 'orange', linewidth=0.5, label='Alpha Cen A')
    ax4.plot(sol0.y[3, :] / AU, sol0.y[4, :] / AU, sol0.y[5, :] / AU, 'blue', linewidth=0.5, label='Alpha Cen B')
    ax4.plot(sol0.y[6, :] / AU, sol0.y[7, :] / AU, sol0.y[8, :] / AU, 'red', linewidth=1.5, label='Proxima')
    ax4.set_xlabel('X (AU)')
    ax4.set_ylabel('Y (AU)')
    ax4.set_zlabel('Z (AU)')
    ax4.set_title('All three stars (relative)')
    ax4.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
