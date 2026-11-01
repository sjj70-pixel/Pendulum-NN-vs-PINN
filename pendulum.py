import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def pendulum_ode(t, y, g=9.81, L=1.0):
    theta, omega = y

    dtheta = omega
    domega = -(g / L) * np.sin(theta)

    return [dtheta, domega]


def generate_data(
    t_end=12.0, train_end=4.0, n_train=50, n_eval=1000, noise_std=0.05, seed=42
):
    # Initial conditions
    theta0 = 1.0
    omega0 = 0.0

    # Solve the differential equation
    sol = solve_ivp(
        pendulum_ode,
        (0, t_end),
        [theta0, omega0],
        dense_output=True,
        rtol=1e-10,
        atol=1e-12,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    # Ground truth over the entire interval
    t_true = np.linspace(0, t_end, n_eval)
    theta_true = sol.sol(t_true)[0]

    # Training observations
    t_train = np.linspace(0, train_end, n_train)
    theta_train_true = sol.sol(t_train)[0]

    # Add Gaussian noise
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_std, size=n_train)
    theta_train = theta_train_true + noise

    return {
        "t_true": t_true,
        "theta_true": theta_true,
        "t_train": t_train,
        "theta_train": theta_train,
    }


if __name__ == "__main__":
    data = generate_data()

    plt.plot(data["t_true"], data["theta_true"], label="Ground truth")

    plt.scatter(
        data["t_train"],
        data["theta_train"],
        color="red",
        s=15,
        label="Noisy observations",
    )

    plt.axvline(4, color="gray", linestyle="--", label="End of training interval")

    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
