import copy
import time
import csv
from pathlib import Path

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

# True: one seed, plots and console output, NO CSV changes.
# False: all seeds, CSV results and summary, plus plots.
SHOW_MODE = True
SHOW_SEED = 42
SEEDS = [42, 43, 44, 45, 46]

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
SAVE_FIGURES = True  # Save plots to figures/; never writes CSVs in Show Mode

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
    active_seeds = [SHOW_SEED] if SHOW_MODE else SEEDS
    print(f"Mode: {'SHOW' if SHOW_MODE else 'RUN'} | Seeds: {active_seeds}")

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

    results = {name: [] for name in experiments}

    # Save the first seed's predictions for the plots in either mode.
    example_predictions = {}
    example_data = None

    # CSV files are created/overwritten ONLY in RUN mode.
    if not SHOW_MODE:
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
                    "Converged",
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
                    "No Energy Converged",
                    "With Energy Converged",
                ]
            )

    # ------------------------------------
    # 1. Train the selected seeds
    # ------------------------------------

    for seed in active_seeds:
        print(f"\n{'=' * 50}")
        print(f"SEED: {seed}")
        print(f"{'=' * 50}")

        torch.manual_seed(seed)

        data = generate_data(
            t_end=T_END,
            train_end=TRAIN_END,
            n_train=N_TRAIN,
            n_eval=N_EVAL,
            noise_std=NOISE_STD,
            seed=seed,
        )

        # Identical initialization within a seed for all models.
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
            # Trainers store a loss every epoch and explicitly flag early convergence.
            epochs_run = history.get("epochs_trained", len(history["total"]))
            converged = history.get("converged")  # None if a trainer does not report it

            results[name].append(
                {
                    "seed": seed,
                    "train_rmse": train_rmse,
                    "test_rmse": test_rmse,
                    "epochs": epochs_run,
                    "time": elapsed,
                    "converged": converged,
                }
            )

            if not SHOW_MODE:
                physics_weight = kwargs.get("lambda_physics", "")
                energy_weight = kwargs.get("lambda_energy", "")

                # Append after EACH completed model, protecting partial results.
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
                            "" if converged is None else converged,
                            physics_weight,
                            energy_weight,
                        ]
                    )

            print(
                f"{name}: "
                f"Train RMSE={train_rmse:.6f}, "
                f"Test RMSE={test_rmse:.6f}, "
                f"Epochs={epochs_run}, "
                f"Converged={converged}, "
                f"Time={elapsed:.2f}s"
            )

            if seed == active_seeds[0]:
                example_predictions[name] = prediction

        if not SHOW_MODE:
            # Direct comparison: add a row after each completed seed.
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
                        (
                            ""
                            if no_energy["converged"] is None
                            else no_energy["converged"]
                        ),
                        (
                            ""
                            if with_energy["converged"] is None
                            else with_energy["converged"]
                        ),
                    ]
                )

        if seed == active_seeds[0]:
            example_data = data

    # ------------------------------------
    # 2. Summary statistics (RUN mode only)
    # ------------------------------------

    if not SHOW_MODE:
        summary = {}

        print("\n" + "=" * 85)
        print("AVERAGED RESULTS")
        print("=" * 85)
        print(
            f"{'Model':<18}"
            f"{'Train RMSE':>20}"
            f"{'Test RMSE':>20}"
            f"{'Time (s)':>20}"
        )

        for name, runs in results.items():
            train_errors = np.array([r["train_rmse"] for r in runs])
            test_errors = np.array([r["test_rmse"] for r in runs])
            times = np.array([r["time"] for r in runs])

            ddof = 1 if len(runs) > 1 else 0
            summary[name] = {
                "train_mean": train_errors.mean(),
                "train_std": train_errors.std(ddof=ddof),
                "test_mean": test_errors.mean(),
                "test_std": test_errors.std(ddof=ddof),
                "time_mean": times.mean(),
                "time_std": times.std(ddof=ddof),
            }

            s = summary[name]
            train_text = f"{s['train_mean']:.4f} ± {s['train_std']:.4f}"
            test_text = f"{s['test_mean']:.4f} ± {s['test_std']:.4f}"
            time_text = f"{s['time_mean']:.1f} ± {s['time_std']:.1f}"
            print(f"{name:<18}{train_text:>20}{test_text:>20}{time_text:>20}")

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
                    "Converged Runs",
                    "Total Runs",
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
                        sum(r["converged"] is True for r in runs),
                        len(runs),
                    ]
                )

    # ------------------------------------
    # 3. Plot predictions for the example seed
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

    ax1.set_title(f"Model predictions: Seed {active_seeds[0]}")
    ax1.set_ylabel("Angle (rad)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Absolute error (rad)")
    plt.tight_layout()
    if SAVE_FIGURES:
        Path("figures").mkdir(exist_ok=True)
        fig.savefig(
            f"figures/predictions_seed_{active_seeds[0]}.png",
            dpi=180,
            bbox_inches="tight",
        )

    # ------------------------------------
    # 4. Plot mean extrapolation RMSE (RUN mode only)
    # ------------------------------------

    if not SHOW_MODE:
        names = list(summary.keys())
        means = [summary[n]["test_mean"] for n in names]
        stds = [summary[n]["test_std"] for n in names]

        fig_rmse, ax_rmse = plt.subplots(figsize=(8, 5))
        ax_rmse.bar(names, means, yerr=stds, capsize=6)
        ax_rmse.set_ylabel("Extrapolation RMSE (rad, log scale)")
        ax_rmse.set_yscale("log")
        ax_rmse.set_title("Mean extrapolation RMSE across seeds")
        ax_rmse.grid(axis="y", alpha=0.3)
        fig_rmse.tight_layout()
        if SAVE_FIGURES:
            Path("figures").mkdir(exist_ok=True)
            fig_rmse.savefig(
                "figures/extrapolation_rmse.png", dpi=180, bbox_inches="tight"
            )

    plt.show()


if __name__ == "__main__":
    main()
