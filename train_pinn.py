import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter


def train_pinn(
    model,
    t_train,
    theta_train,
    t_end=12.0,
    g=9.81,
    L=1.0,
    epochs=100000,
    lr=1e-3,
    lambda_physics=0.01,
    lambda_ic=1.0,
    lambda_energy=0.1,
    n_collocation=200,
    collocation_method="fixed",
    successful_checks=0,
    check_interval=500,
    required_checks=5,
):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Convert observations to tensors
    t_train = torch.as_tensor(t_train, dtype=dtype, device=device).reshape(-1, 1)

    theta_train = torch.as_tensor(theta_train, dtype=dtype, device=device).reshape(
        -1, 1
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    history = {"total": [], "data": [], "physics": [], "ic": [], "energy": []}
    history["converged"] = False

    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()

        # 1. Data loss
        theta_pred = model(t_train)
        loss_data = mse(theta_pred, theta_train)

        # 2. Initial condition loss
        t0 = torch.tensor([[0.0]], dtype=dtype, device=device, requires_grad=True)

        theta0_pred = model(t0)

        omega0_pred = torch.autograd.grad(
            theta0_pred,
            t0,
            grad_outputs=torch.ones_like(theta0_pred),
            create_graph=True,
        )[0]

        loss_ic = (theta0_pred - 1.0).square().mean()
        loss_ic += omega0_pred.square().mean()

        # 3. Physics loss
        if collocation_method == "fixed":
            t_physics = torch.linspace(
                0, t_end, n_collocation, dtype=dtype, device=device
            ).reshape(-1, 1)

        elif collocation_method == "random":
            t_physics = t_end * torch.rand(n_collocation, 1, dtype=dtype, device=device)

        else:
            raise ValueError("Unknown collocation method")

        t_physics = t_physics.requires_grad_(True)

        theta_physics = model(t_physics)

        # First derivative: angular velocity
        omega_physics = torch.autograd.grad(
            theta_physics,
            t_physics,
            grad_outputs=torch.ones_like(theta_physics),
            create_graph=True,
        )[0]

        # Second derivative: angular acceleration
        alpha_physics = torch.autograd.grad(
            omega_physics,
            t_physics,
            grad_outputs=torch.ones_like(omega_physics),
            create_graph=True,
        )[0]

        # 4. Energy conservation loss
        # Initial energy (theta0=1, omega0=0)
        energy0 = g * L * (1 - torch.cos(torch.tensor(1.0, dtype=dtype, device=device)))

        # Predicted energy at collocation points
        energy = 0.5 * L**2 * omega_physics**2 + g * L * (1 - torch.cos(theta_physics))

        # Normalized energy loss
        loss_energy = ((energy - energy0) / energy0).square().mean()

        # Pendulum equation residual
        residual = alpha_physics + (g / L) * torch.sin(theta_physics)

        loss_physics = residual.square().mean()

        # 4. Total loss
        loss = (
            loss_data
            + lambda_physics * loss_physics
            + lambda_ic * loss_ic
            + lambda_energy * loss_energy
        )

        # Record loss history
        history["total"].append(loss.item())
        history["data"].append(loss_data.item())
        history["physics"].append(loss_physics.item())
        history["ic"].append(loss_ic.item())
        history["energy"].append(loss_energy.item())

        # Check if the model has passed learning thresholds
        if (epoch + 1) % check_interval == 0:

            converged = (
                loss_data.item() < 0.003
                and loss_physics.item() < 0.01
                and loss_ic.item() < 1e-4
                and loss_energy.item() < 0.001
            )

            if converged:
                successful_checks += 1
            else:
                successful_checks = 0

            if successful_checks >= required_checks:
                history["converged"] = True
                print(f"Converged at epoch {epoch + 1}")
                break

        # 5. Backpropagation
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 500 == 0:
            print(
                f"Epoch {epoch+1}: "
                f"Total={loss.item():.6f}, "
                f"Data={loss_data.item():.6f}, "
                f"Physics={loss_physics.item():.6f}, "
                f"IC={loss_ic.item():.6f}"
                f"Energy={loss_energy.item():.6f}"
            )
    history["epochs_trained"] = len(history["total"])
    model.eval()

    return model, history


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from pendulum import generate_data
    from model import PendulumNet

    torch.manual_seed(42)

    data = generate_data(t_end=12)
    model = PendulumNet()

    model, history = train_pinn(
        model, data["t_train"], data["theta_train"], collocation_method="fixed"
    )

    # Evaluate over the full 12 seconds
    t = torch.tensor(data["t_true"], dtype=torch.float32).reshape(-1, 1)

    with torch.no_grad():
        theta_pred = model(t).cpu().numpy().flatten()

    # Calculate RMSE
    t_true = data["t_true"]
    theta_true = data["theta_true"]

    train_mask = t_true <= 4.0
    test_mask = t_true > 4.0

    rmse_train = np.sqrt(
        np.mean((theta_pred[train_mask] - theta_true[train_mask]) ** 2)
    )

    rmse_test = np.sqrt(np.mean((theta_pred[test_mask] - theta_true[test_mask]) ** 2))

    print(f"Training RMSE: {rmse_train:.6f}")
    print(f"Extrapolation RMSE: {rmse_test:.6f}")

    # Plot predictions
    plt.plot(t_true, theta_true, label="Ground truth")
    plt.plot(t_true, theta_pred, label="PINN prediction")

    plt.scatter(
        data["t_train"], data["theta_train"], s=15, color="red", label="Observations"
    )

    plt.axvline(4, color="gray", linestyle="--", label="End of training interval")

    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    fig, ax = plt.subplots()

    for key in ["data", "physics", "ic", "energy"]:
        ax.plot(history[key], label=key)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Unweighted loss")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.show()

    plt.figure(figsize=(10, 5))

    for key in ["data", "physics", "energy"]:
        values = np.array(history[key])

        smooth = median_filter(values, size=501, mode="nearest")

    plt.semilogy(smooth, label=key)

    plt.xlabel("Epoch")
    plt.ylabel("Smoothed loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
