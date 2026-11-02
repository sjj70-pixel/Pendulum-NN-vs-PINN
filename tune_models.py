import numpy as np
import torch
import optuna

from pendulum import generate_data
from model import PendulumNet
from train_nn import train_nn
from train_pinn import train_pinn

# ------------------------------------
# Settings
# ------------------------------------

SEED = 42
N_TRIALS = {"NN": 8, "PINN": 20}
EPOCHS = 10000

T_END = 12.0
TRAIN_END = 4.0
N_COLLOCATION = 200


def objective(trial, model_type, data, train_idx, val_idx):

    # Hyperparameters
    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)

    # Same initial network weights for every trial
    torch.manual_seed(SEED)
    model = PendulumNet()

    t_train = data["t_train"][train_idx]
    theta_train = data["theta_train"][train_idx]

    # ------------------------------------
    # Train the selected model
    # ------------------------------------

    if model_type == "NN":

        model, history = train_nn(
            model, t_train, theta_train, epochs=EPOCHS, lr=lr, lambda_ic=1.0
        )

    elif model_type == "PINN":

        lambda_physics = trial.suggest_float("lambda_physics", 1e-3, 1e-1, log=True)

        lambda_energy = trial.suggest_float("lambda_energy", 1e-2, 1.0, log=True)

        model, history = train_pinn(
            model,
            t_train,
            theta_train,
            t_end=T_END,
            epochs=EPOCHS,
            lr=lr,
            lambda_physics=lambda_physics,
            lambda_energy=lambda_energy,
            lambda_ic=1.0,
            n_collocation=N_COLLOCATION,
        )

    # ------------------------------------
    # Validation using held-out observations
    # ------------------------------------

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    t_val = torch.as_tensor(
        data["t_train"][val_idx], dtype=dtype, device=device
    ).reshape(-1, 1)

    theta_val = data["theta_train"][val_idx]

    model.eval()

    with torch.no_grad():
        prediction = model(t_val).cpu().numpy().flatten()

    val_rmse = np.sqrt(np.mean((prediction - theta_val) ** 2))

    return float(val_rmse)


def run_study(model_type, data, train_idx, val_idx):

    study = optuna.create_study(
        direction="minimize",
        study_name=f"pendulum_{model_type}_observed_val",
        storage="sqlite:///observed_validation.db",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=3),
    )

    # Start with our previous configuration
    if len(study.trials) == 0:

        baseline = {"lr": 1e-3}

        if model_type == "PINN":
            baseline.update({"lambda_physics": 0.01, "lambda_energy": 0.1})

        study.enqueue_trial(baseline)

    remaining = max(0, N_TRIALS[model_type] - len(study.trials))

    if remaining > 0:
        study.optimize(
            lambda trial: objective(trial, model_type, data, train_idx, val_idx),
            n_trials=remaining,
        )

    print(f"\nBest {model_type} validation RMSE:")
    print(study.best_value)

    print(f"\nBest {model_type} parameters:")
    print(study.best_params)

    study.trials_dataframe().to_csv(f"tuning_{model_type.lower()}.csv", index=False)

    return study


if __name__ == "__main__":

    # Generate the same noisy dataset for both models
    data = generate_data(
        t_end=T_END,
        train_end=TRAIN_END,
        n_train=50,
        n_eval=1000,
        noise_std=0.05,
        seed=SEED,
    )

    # Validate every fifth observation
    val_idx = np.arange(4, 50, 5)

    train_idx = np.setdiff1d(np.arange(50), val_idx)

    print("Training observations:", len(train_idx))
    print("Validation observations:", len(val_idx))

    # Tune both models on exactly the same split
    nn_study = run_study("NN", data, train_idx, val_idx)

    pinn_study = run_study("PINN", data, train_idx, val_idx)

    print("\n" + "=" * 50)
    print("FINAL TUNING RESULTS")
    print("=" * 50)

    print("\nNN:")
    print(nn_study.best_params)
    print("Validation RMSE:", nn_study.best_value)

    print("\nPINN:")
    print(pinn_study.best_params)
    print("Validation RMSE:", pinn_study.best_value)
