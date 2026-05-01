import matplotlib.pyplot as plt
import numpy as np

from gravity_nbody.physics_engine import PhysicsEngine


def main():
    # Configuration
    n_bodies = 30
    units = 'SI'

    # Time settings
    year_seconds = 365.25 * 24 * 3600
    t_end = 2 * year_seconds  # Simulate for 2 years
    t_points = 10000
    t_span = (0, t_end)
    t_eval = np.linspace(0, t_end, t_points)

    # Initial Conditions (Random but scaled for visibility)
    pos0 = (np.random.rand(n_bodies, 3) - 0.5) * 2 * 1.5e11
    vel0 = (np.random.rand(n_bodies, 3) - 0.5) * 2 * 30000
    masses = np.random.rand(n_bodies) * 1.989e30

    state = np.hstack((pos0.flatten(), vel0.flatten()))

    engine = PhysicsEngine(units=units)

    print("Running simulation...")
    sol = engine.run_simulation(state, t_span, masses, t_eval=t_eval)

    # Reshape positions for easier plotting: (Time, Body, Coordinate)
    positions = sol.y[:3 * n_bodies].T.reshape(-1, n_bodies, 3)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.jet(np.linspace(0, 1, n_bodies))

    for i in range(n_bodies):
        ax.plot(
            positions[:, i, 0],
            positions[:, i, 1],
            positions[:, i, 2],
            color=colors[i],
            linewidth=1.2,
            alpha=0.8,
            label=f'Body {i + 1}',
        )

        # Mark final position only
        ax.scatter(
            positions[-1, i, 0],
            positions[-1, i, 1],
            positions[-1, i, 2],
            color=colors[i],
            s=200,
            marker='o',
            edgecolors='black',
            linewidths=2,
            zorder=10,
        )

    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_zlabel('Z (m)', fontsize=11)
    ax.set_title(f'{n_bodies}-Body Gravitational Simulation', fontsize=13, fontweight='bold', pad=15)

    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.legend(fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
