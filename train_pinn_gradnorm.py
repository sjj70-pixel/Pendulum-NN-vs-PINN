import math
import torch
import torch.nn as nn


def train_pinn_gradnorm(
    model,
    t_train,
    theta_train,
    t_end=12.0,
    g=9.81,
    L=1.0,
    epochs=10000,
    lr=1e-3,
    lr_weights=1e-2,
    alpha=0.5,
    lambda_ic=1.0,
    n_collocation=200,
):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    t_train = torch.as_tensor(t_train, dtype=dtype, device=device).reshape(-1, 1)

    theta_train = torch.as_tensor(theta_train, dtype=dtype, device=device).reshape(
        -1, 1
    )

    # Optimizer for neural network parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Initial loss weights: data, physics, energy
    initial_weights = torch.tensor([1.0, 0.01, 0.1], dtype=dtype, device=device)

    weight_sum = initial_weights.sum()

    # Trainable weight parameters
    logits = nn.Parameter(initial_weights.log())

    weight_optimizer = torch.optim.Adam([logits], lr=lr_weights)

    # Measure gradients using the final linear layer
    reference = model.network[-1].weight

    initial_losses = None

    history = {
        "total": [],
        "data": [],
        "physics": [],
        "energy": [],
        "ic": [],
        "weight_data": [],
        "weight_physics": [],
        "weight_energy": [],
    }

    energy0 = g * L * (1 - math.cos(1.0))

    model.train()

    for epoch in range(epochs):

        # ------------------------------------
        # 1. Calculate the individual losses
        # ------------------------------------

        # Data loss
        theta_pred = model(t_train)
        loss_data = (theta_pred - theta_train).square().mean()

        # Initial condition loss
        t0 = torch.tensor([[0.0]], dtype=dtype, device=device, requires_grad=True)

        theta0 = model(t0)

        omega0 = torch.autograd.grad(theta0, t0, create_graph=True)[0]

        loss_ic = (theta0 - 1.0).square().mean() + omega0.square().mean()

        # Collocation points
        t_physics = (
            torch.linspace(0, t_end, n_collocation, dtype=dtype, device=device)
            .reshape(-1, 1)
            .requires_grad_(True)
        )

        theta = model(t_physics)

        omega = torch.autograd.grad(
            theta, t_physics, grad_outputs=torch.ones_like(theta), create_graph=True
        )[0]

        acceleration = torch.autograd.grad(
            omega, t_physics, grad_outputs=torch.ones_like(omega), create_graph=True
        )[0]

        # Physics loss
        residual = acceleration + (g / L) * torch.sin(theta)
        loss_physics = residual.square().mean()

        # Energy loss
        energy = 0.5 * L**2 * omega.square() + g * L * (1 - torch.cos(theta))

        loss_energy = ((energy - energy0) / energy0).square().mean()

        # Losses controlled by GradNorm
        task_losses = torch.stack([loss_data, loss_physics, loss_energy])

        if initial_losses is None:
            initial_losses = task_losses.detach().clamp_min(1e-12)

        # ------------------------------------
        # 2. Calculate gradient magnitudes
        # ------------------------------------

        grad_norms = []

        for task_loss in task_losses:
            grad = torch.autograd.grad(task_loss, reference, retain_graph=True)[0]

            grad_norms.append(grad.detach().norm(p=2))

        grad_norms = torch.stack(grad_norms)

        # ------------------------------------
        # 3. GradNorm weight update
        # ------------------------------------

        # Positive weights with fixed total
        weights = weight_sum * torch.softmax(logits, dim=0)

        weighted_grads = weights * grad_norms

        # Relative inverse training rates
        with torch.no_grad():
            loss_ratios = task_losses.detach() / initial_losses

            rates = loss_ratios / loss_ratios.mean().clamp_min(1e-12)

            target_grads = weighted_grads.detach().mean() * rates.pow(alpha)

        # Optimize the loss weights
        gradnorm_loss = (weighted_grads - target_grads).abs().sum()

        weight_optimizer.zero_grad()
        gradnorm_loss.backward()
        weight_optimizer.step()

        # Recalculate the updated weights
        weights = (weight_sum * torch.softmax(logits, dim=0)).detach()

        # ------------------------------------
        # 4. Update the neural network
        # ------------------------------------

        loss = (weights * task_losses).sum() + lambda_ic * loss_ic

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ------------------------------------
        # 5. Record history
        # ------------------------------------

        history["total"].append(loss.item())
        history["data"].append(loss_data.item())
        history["physics"].append(loss_physics.item())
        history["energy"].append(loss_energy.item())
        history["ic"].append(loss_ic.item())

        history["weight_data"].append(weights[0].item())
        history["weight_physics"].append(weights[1].item())
        history["weight_energy"].append(weights[2].item())

        if (epoch + 1) % 500 == 0:
            print(
                f"Epoch {epoch+1}: "
                f"Total={loss.item():.6f}, "
                f"Data={loss_data.item():.6f}, "
                f"Physics={loss_physics.item():.6f}, "
                f"Energy={loss_energy.item():.6f}"
            )

            print(
                f"  Weights: "
                f"Data={weights[0].item():.5f}, "
                f"Physics={weights[1].item():.5f}, "
                f"Energy={weights[2].item():.5f}"
            )

    model.eval()

    return model, history


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from pendulum import generate_data
    from model import PendulumNet

    torch.manual_seed(42)

    data = generate_data()
    model = PendulumNet()

    model, history = train_pinn_gradnorm(model, data["t_train"], data["theta_train"])

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

    plt.figure()

    for name in ["data", "physics", "energy"]:
        plt.plot(history[f"weight_{name}"], label=name)

    plt.xlabel("Epoch")
    plt.ylabel("GradNorm weight")
    plt.yscale("log")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
