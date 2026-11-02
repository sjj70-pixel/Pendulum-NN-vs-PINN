import copy
import time

import numpy as np
import torch
import matplotlib.pyplot as plt

from pendulum import generate_data
from model import PendulumNet
from train_nn import train_nn
from train_pinn import train_pinn
from train_pinn_gradnorm import train_pinn_gradnorm

# ------------------------------------
# Experiment settings
# ------------------------------------

# SEEDS = [42, 43, 44, 45, 46]
SEEDS = [42]

T_END = 12.0
TRAIN_END = 4.0
N_TRAIN = 50
N_EVAL = 1000
NOISE_STD = 0.05

EPOCHS = 10000
LEARNING_RATE = 1e-3

LAMBDA_IC = 1.0
LAMBDA_PHYSICS = 0.01
LAMBDA_ENERGY = 0.1

N_COLLOCATION = 200
RUN_GRADNORM = True


def evaluate_model(model, data):
    """Predict trajectory and calculate RMSE."""

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    t = torch.as_tensor(data["t_true"], dtype=dtype, device=device).reshape(-1, 1)

    model.eval()

    with torch.no_grad():
        prediction = model(t).cpu().numpy().flatten()

    t_true = data["t_true"]
    theta_true = data["theta_true"]

    train_mask = t_true <= TRAIN_END
    test_mask = t_true > TRAIN_END

    train_rmse = np.sqrt(
        np.mean((prediction[train_mask] - theta_true[train_mask]) ** 2)
    )

    test_rmse = np.sqrt(np.mean((prediction[test_mask] - theta_true[test_mask]) ** 2))

    return prediction, train_rmse, test_rmse


def main():

    experiments = {
        "NN": (train_nn, {"lambda_ic": LAMBDA_IC}),
        "PINN": (
            train_pinn,
            {
                "t_end": T_END,
                "lambda_ic": LAMBDA_IC,
                "lambda_physics": LAMBDA_PHYSICS,
                "lambda_energy": LAMBDA_ENERGY,
                "n_collocation": N_COLLOCATION,
            },
        ),
    }

    if RUN_GRADNORM:
        experiments["PINN GradNorm"] = (
            train_pinn_gradnorm,
            {
                "t_end": T_END,
                "lambda_ic": LAMBDA_IC,
                "n_collocation": N_COLLOCATION,
                "alpha": 0.5,
                "lr_weights": 1e-2,
            },
        )

    # Store results for every seed
    results = {name: [] for name in experiments}

    # Save seed 42 predictions for visualization
    example_predictions = {}
    example_data = None

    # ------------------------------------
    # 1. Run all seeds
    # ------------------------------------

    for seed in SEEDS:

        print(f"\n{'='*50}")
        print(f"SEED: {seed}")
        print(f"{'='*50}")

        torch.manual_seed(seed)

        data = generate_data(
            t_end=T_END,
            train_end=TRAIN_END,
            n_train=N_TRAIN,
            n_eval=N_EVAL,
            noise_std=NOISE_STD,
            seed=seed,
        )

        # Identical initialization within this seed
        base_model = PendulumNet()

        for name, (trainer, kwargs) in experiments.items():

            print(f"\nTraining {name}...")

            model = copy.deepcopy(base_model)

            start = time.perf_counter()

            model, history = trainer(
                model,
                data["t_train"],
                data["theta_train"],
                epochs=EPOCHS,
                lr=LEARNING_RATE,
                **kwargs,
            )

            elapsed = time.perf_counter() - start

            prediction, train_rmse, test_rmse = evaluate_model(model, data)

            results[name].append(
                {
                    "seed": seed,
                    "train_rmse": train_rmse,
                    "test_rmse": test_rmse,
                    "time": elapsed,
                }
            )

            print(
                f"{name}: "
                f"Train RMSE={train_rmse:.6f}, "
                f"Test RMSE={test_rmse:.6f}, "
                f"Time={elapsed:.2f}s"
            )

            if seed == SEEDS[0]:
                example_predictions[name] = prediction

        if seed == SEEDS[0]:
            example_data = data

    # ------------------------------------
    # 2. Calculate summary statistics
    # ------------------------------------

    summary = {}

    print("\n" + "=" * 85)
    print("AVERAGED RESULTS")
    print("=" * 85)

    print(
        f"{'Model':<18}" f"{'Train RMSE':>20}" f"{'Test RMSE':>20}" f"{'Time (s)':>20}"
    )

    for name, runs in results.items():

        train_errors = np.array([r["train_rmse"] for r in runs])

        test_errors = np.array([r["test_rmse"] for r in runs])

        times = np.array([r["time"] for r in runs])

        summary[name] = {
            "train_mean": train_errors.mean(),
            "train_std": train_errors.std(ddof=1),
            "test_mean": test_errors.mean(),
            "test_std": test_errors.std(ddof=1),
            "time_mean": times.mean(),
            "time_std": times.std(ddof=1),
        }

        s = summary[name]

        train_text = f"{s['train_mean']:.4f} ± {s['train_std']:.4f}"

        test_text = f"{s['test_mean']:.4f} ± {s['test_std']:.4f}"

        time_text = f"{s['time_mean']:.1f} ± {s['time_std']:.1f}"

        print(f"{name:<18}" f"{train_text:>20}" f"{test_text:>20}" f"{time_text:>20}")

    # ------------------------------------
    # 3. Plot predictions for first seed
    # ------------------------------------

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    t = example_data["t_true"]
    theta_true = example_data["theta_true"]

    ax1.plot(t, theta_true, color="black", label="Ground truth")

    ax1.scatter(
        example_data["t_train"],
        example_data["theta_train"],
        color="red",
        s=12,
        label="Observations",
    )

    for name, prediction in example_predictions.items():

        ax1.plot(t, prediction, label=name)

        error = np.abs(prediction - theta_true)
        ax2.plot(t, error, label=name)

    for ax in (ax1, ax2):
        ax.axvline(TRAIN_END, color="gray", linestyle="--")
        ax.legend()
        ax.grid(alpha=0.3)

    ax1.set_title(f"Model predictions: Seed {SEEDS[0]}")
    ax1.set_ylabel("Angle (rad)")

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Absolute error (rad)")

    plt.tight_layout()

    # ------------------------------------
    # 4. Plot averaged extrapolation RMSE
    # ------------------------------------

    names = list(summary.keys())

    means = [summary[n]["test_mean"] for n in names]
    stds = [summary[n]["test_std"] for n in names]

    plt.figure(figsize=(8, 5))

    plt.bar(names, means, yerr=stds, capsize=6)

    plt.ylabel("Extrapolation RMSE (rad)")
    plt.title("Mean extrapolation RMSE across seeds")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
