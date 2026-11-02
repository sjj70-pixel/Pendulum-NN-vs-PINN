import torch
import torch.nn as nn


def train_nn(model, t_train, theta_train, epochs=10000, lr=1e-3, lambda_ic=1.0):
    # Convert data to PyTorch tensors
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    t_train = torch.as_tensor(t_train, dtype=dtype, device=device).reshape(-1, 1)

    theta_train = torch.as_tensor(theta_train, dtype=dtype, device=device).reshape(
        -1, 1
    )

    # Initial time, with gradients enabled
    t0 = torch.tensor([[0.0]], dtype=dtype, device=device, requires_grad=True)

    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    # Store training history
    history = {"total": [], "data": [], "ic": []}

    check_interval = 500
    required_checks = 5
    successful_checks = 0

    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()

        # 1. Data loss
        theta_pred = model(t_train)
        loss_data = mse(theta_pred, theta_train)

        # 2. Initial condition loss
        theta0_pred = model(t0)

        omega0_pred = torch.autograd.grad(
            theta0_pred,
            t0,
            grad_outputs=torch.ones_like(theta0_pred),
            create_graph=True,
        )[0]

        loss_ic = (theta0_pred - 1.0).square().mean()
        loss_ic += omega0_pred.square().mean()

        # 3. Total loss
        loss = loss_data + lambda_ic * loss_ic

        # 4. Backpropagation
        loss.backward()
        optimizer.step()

        # Record losses
        history["total"].append(loss.item())
        history["data"].append(loss_data.item())
        history["ic"].append(loss_ic.item())

        if (epoch + 1) % check_interval == 0:
            print(
                f"Epoch {epoch+1}: "
                f"Total={loss.item():.6f}, "
                f"Data={loss_data.item():.6f}, "
                f"IC={loss_ic.item():.6f}"
            )

        # Check for convergence
        if (epoch + 1) % check_interval == 0:

            converged = loss_data.item() < 0.003 and loss_ic.item() < 1e-4

            if converged:
                successful_checks += 1
            else:
                successful_checks = 0

            if successful_checks >= required_checks:
                print(f"NN converged at epoch {epoch + 1}")
                break

    model.eval()

    return model, history


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pendulum import generate_data
    from model import PendulumNet

    torch.manual_seed(42)

    data = generate_data()
    model = PendulumNet()

    model, history = train_nn(model, data["t_train"], data["theta_train"])

    # Evaluate over the full 12 seconds
    t = torch.tensor(data["t_true"], dtype=torch.float32).reshape(-1, 1)

    with torch.no_grad():
        theta_pred = model(t).numpy().flatten()

    plt.plot(data["t_true"], data["theta_true"], label="Ground truth")
    plt.plot(data["t_true"], theta_pred, label="NN prediction")
    plt.scatter(
        data["t_train"], data["theta_train"], s=15, color="red", label="Observations"
    )
    plt.axvline(4, color="gray", linestyle="--")

    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
