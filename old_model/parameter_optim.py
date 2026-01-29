import json

from dataset import *
import torch
from basic_inpainting import *
import os
import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms


STUDY_NAME = "inpainting_hyperparam_optimization"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCHSIZE = 16
EPOCHS = 10
N_TRAIN_EXAMPLES = BATCHSIZE * 30
N_VALID_EXAMPLES = BATCHSIZE * 10

def objective(trial):
    # --- arch hyperparameters ---
    base_chan = trial.suggest_categorical("base_channels", [16, 32, 48, 64])
    depth = trial.suggest_int("enc_depth", 2, 4)
    encoder_channels = [base_chan * (2 ** i) for i in range(depth)]
    latent_dim = trial.suggest_categorical("latent_dim", [128, 256, 512, 1024])
    num_inpaint = trial.suggest_int("num_inpainting_layers", 1, 4)
    use_hardswish = trial.suggest_categorical("use_hardswish", [True, False])

    # --- optimizer / lr ---
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

    # build model with sampled architecture
    model = BasicInpainting(
        encoder_channels=tuple(encoder_channels),
        latent_dim=latent_dim,
        num_inpainting_layers=num_inpaint,
        use_hardswish=use_hardswish,
    ).to(DEVICE)

    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    ds = ImageMaskDataset(images_dir='../output/images', masks_dir='../output/masks')
    train_ds, valid_ds, test_ds = ds.split_ds(train_ratio=0.8, test_ratio=0.1, valid_ratio=0.1)
    train_loader = DataLoader(train_ds, batch_size=BATCHSIZE, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCHSIZE, shuffle=False)

    loss_fn = nn.L1Loss()
    last_val_loss = float("inf")

    for epoch in range(EPOCHS):
        train_loss = model.train_one_epoch(train_loader, optimizer, loss_fn, DEVICE)
        val_loss = model.evaluate(valid_loader, loss_fn, DEVICE)
        last_val_loss = val_loss

        # report to Optuna (study.direction = "maximize" - map lower loss -> higher objective)
        trial.report(-val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # store sampled arch params for best trial outside (Optuna stores trial.params automatically)
    return -last_val_loss

def on_trial_complete(study, trial):
    """Callback zapisywający najlepszy trial po każdym zakończeniu trialu."""
    try:
        best = study.best_trial if study.best_trial is not None else trial

        # zapisz czytelny plik tekstowy (istniejąca funkcja można użyć lub nadpisać)
        save_params_to_file(best, filename="best_params.txt")

        # zapisz JSON z wartością i parametrami atomowo
        data = {"value": best.value, "params": best.params}
        tmp = "best_trial.tmp.json"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, "best_trial.json")
    except Exception as e:
        print("Warning: failed to save best trial:", e)

def save_params_to_file(trial, filename="best_params.txt"):
    with open(filename, "w") as f:
        for key, value in trial.params.items():
            f.write(f"{key}: {value}\n")

if __name__ == "__main__":
    print(DEVICE)
    study = optuna.create_study(direction="maximize", study_name=STUDY_NAME, load_if_exists=True)
    study.optimize(objective, n_trials=20, timeout=1200, gc_after_trial=True, n_jobs=4, callbacks=[on_trial_complete])
    pruned = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    print("Best trial value:", study.best_trial.value)
    save_params_to_file(study.best_trial, filename="best_params.txt")