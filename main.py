import copy
import time
import csv

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

SEEDS = [42, 43, 44, 45, 46]
# SEEDS = [42]

T_END = 12.0
TRAIN_END = 4.0
N_TRAIN = 50
N_EVAL = 1000
NOISE_STD = 0.05

EPOCHS = 100000
LEARNING_RATE = 1e-3

LAMBDA_IC = 1.0
LAMBDA_PHYSICS = 0.01
LAMBDA_ENERGY = 0.1

N_COLLOCATION = 200
RUN_GRADNORM = False

RESULTS_CSV = "results_12s.csv"
COMPARISON_CSV = "energy_comparison_12s.csv"
SUMMARY_CSV = "summary_12s.csv"


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
        "PINN (no energy)": (
            train_pinn,
            {
                "t_end": T_END,
                "lambda_ic": LAMBDA_IC,
                "lambda_physics": LAMBDA_PHYSICS,
                "lambda_energy": 0.0,
                "n_collocation": N_COLLOCATION,
            },
        ),
        "PINN (energy)": (
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

    # Save to csv
    # Initialize CSV files
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Model",
                "Seed",
                "Train RMSE",
                "Test RMSE",
                "Epochs",
                "Training Time (s)",
                "Physics Weight",
                "Energy Weight",
            ]
        )

    with open(COMPARISON_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Seed",
                "No Energy Test RMSE",
                "With Energy Test RMSE",
                "RMSE Improvement",
                "No Energy Epochs",
                "With Energy Epochs",
                "Epochs Saved",
                "No Energy Time (s)",
                "With Energy Time (s)",
            ]
        )

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

            # Actual epochs trained, including early stopping
            epochs_run = len(history["total"])

            results[name].append(
                {
                    "seed": seed,
                    "train_rmse": train_rmse,
                    "test_rmse": test_rmse,
                    "epochs": epochs_run,
                    "time": elapsed,
                }
            )

            # Fixed loss weights, if applicable
            physics_weight = kwargs.get("lambda_physics", "")
            energy_weight = kwargs.get("lambda_energy", "")

            # Save immediately after each completed model
            with open(RESULTS_CSV, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        name,
                        seed,
                        train_rmse,
                        test_rmse,
                        epochs_run,
                        elapsed,
                        physics_weight,
                        energy_weight,
                    ]
                )

            print(
                f"{name}: "
                f"Train RMSE={train_rmse:.6f}, "
                f"Test RMSE={test_rmse:.6f}, "
                f"Epochs={epochs_run}, "
                f"Time={elapsed:.2f}s"
            )

            if seed == SEEDS[0]:
                example_predictions[name] = prediction

        # Compare PINN with and without energy
        no_energy = results["PINN (no energy)"][-1]
        with_energy = results["PINN (energy)"][-1]

        rmse_improvement = no_energy["test_rmse"] - with_energy["test_rmse"]

        epochs_saved = no_energy["epochs"] - with_energy["epochs"]

        with open(COMPARISON_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    seed,
                    no_energy["test_rmse"],
                    with_energy["test_rmse"],
                    rmse_improvement,
                    no_energy["epochs"],
                    with_energy["epochs"],
                    epochs_saved,
                    no_energy["time"],
                    with_energy["time"],
                ]
            )

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

    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "Model",
                "Mean Train RMSE",
                "Std Train RMSE",
                "Mean Test RMSE",
                "Std Test RMSE",
                "Mean Epochs",
                "Std Epochs",
                "Mean Time (s)",
                "Std Time (s)",
            ]
        )

        for name, runs in results.items():
            train = np.array([r["train_rmse"] for r in runs])
            test = np.array([r["test_rmse"] for r in runs])
            epochs = np.array([r["epochs"] for r in runs])
            times = np.array([r["time"] for r in runs])

            ddof = 1 if len(runs) > 1 else 0

            writer.writerow(
                [
                    name,
                    train.mean(),
                    train.std(ddof=ddof),
                    test.mean(),
                    test.std(ddof=ddof),
                    epochs.mean(),
                    epochs.std(ddof=ddof),
                    times.mean(),
                    times.std(ddof=ddof),
                ]
            )

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
