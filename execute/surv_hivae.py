import numpy as np
import torch
import torch.optim as optim
import time
import time
import pandas as pd
import importlib
import random
import warnings
import os
import matplotlib.pyplot as plt
import seaborn as sns


import sys
from pathlib import Path
module_path = Path.cwd().parent / 'utils'
sys.path.append(str(module_path))
import data_processing, visualization, statistic, metrics, likelihood, theta_estimation
from data_processing import MyCustomDataset
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator


warnings.filterwarnings("ignore")

def set_seed(seed=1):
    random.seed(seed)                            # Python built-in
    np.random.seed(seed)                         # NumPy
    torch.manual_seed(seed)                      # PyTorch (CPU)


# ───────────────────────
# TRAINING HIVAE FUNCTION
# ───────────────────────
def train_HIVAE(vae_model, data, miss_mask, true_miss_mask, feat_types_dict, batch_size, lr, epochs, verbose = True, seed=1, start_epoch=0, norm_mode="global", differential_privacy=False, target_epsilon=None, target_delta=1e-5, max_grad_norm=1.0):
    """ 
        HIVAE training function.
        Splits `data` into 90/10 train/validation, freezes per-feature normalization statistics on the full training set and stores them on
        `vae_model.feat_normalization_globals`, then runs the optimizer for `epochs` with early stopping on the validation ELBO. The `seed` argument
        controls the split RNG and the rest of the loop's stochasticity. 
        This version now includes the batch-corrected setup (before train_HIVAE_bis) and the differentially-private setup (before train_HIVAE_DP).

        `norm_mode` selects how the encoder input is standardized (see data_processing.resolve_feat_normalization_globals):
            "global" (default) : frozen training-set statistics for every feature — every batch uses the same scale.
            "batch"            : every feature recomputes its statistics from each batch (mean/var for real / pos / legacy 'surv',
                                min/max for the survival families). Not allowed under differential privacy.
    """
    _validate_norm_mode(norm_mode, differential_privacy=True)

    if differential_privacy:
        # ─── Opacus compatibility check ────────────────────────────────────
        # Catches unsupported layers (BatchNorm, LSTM, …) before training.
        errors = ModuleValidator.validate(vae_model, strict=False)
        if errors:
            raise ValueError(
                "vae_model is not Opacus-compatible. Offending layers:\n  - "
                + "\n  - ".join(str(e) for e in errors)
            )

    # ─── Seed every source of randomness for reproducibility ────────────
    rng = np.random.default_rng(seed=seed)
    torch.manual_seed(seed)
    noise_generator = torch.Generator(device="cpu").manual_seed(seed)

    # ─── Train-test split on control ────────────
    train_test_share = .9
    n_samples = data.shape[0]
    n_train_samples = int(train_test_share * n_samples)
    train_index = rng.choice(n_samples, n_train_samples, replace=False)
    test_index = np.setdiff1d(np.arange(n_samples), train_index)

    data_train = data[train_index]
    miss_mask_train = miss_mask[train_index]
    true_miss_mask_train = true_miss_mask[train_index]

    data_test = data[test_index]
    miss_mask_test = miss_mask[test_index]
    true_miss_mask_test = true_miss_mask[test_index]

    # ─── Number of batches ────────────
    n_train_samples = data_train.shape[0]
    if n_train_samples < batch_size:
        raise ValueError("Batch size must be less than the number of training samples")

    # ─── Compute real missing mask ────────────
    miss_mask_train = torch.multiply(miss_mask_train, true_miss_mask_train)
    miss_mask_test = torch.multiply(miss_mask_test, true_miss_mask_test)

    # ─── Test/val batching ────────────
    n_test_samples = data_test.shape[0]
    batch_test_size = n_test_samples
    n_batches_test = int(np.floor(n_test_samples / batch_test_size))

    # ─── Freeze per-feature normalization stats on the training set ───
    # Full global stats are always computed; resolve_feat_normalization_globals
    # then either keeps them all (norm_mode="global") or drops every entry to
    # None so all features fall back to per-batch statistics (norm_mode="batch").
    _feat_globals = data_processing.compute_feat_normalization_globals(
        data_train, feat_types_dict, miss_mask_train
    )
    vae_model.feat_normalization_globals = data_processing.resolve_feat_normalization_globals(
        _feat_globals, feat_types_dict, norm_mode
    )

    # ─── Optimizer, dataset, loader ────────────────────────────────────
    optimizer = optim.Adam(vae_model.parameters(), lr=lr)
    dataset = MyCustomDataset(data_train, miss_mask_train, feat_types_dict)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) 

    if differential_privacy:
        # ─── Wrap with Opacus, calibrating σ to (target_epsilon, target_delta) ──
        privacy_engine = PrivacyEngine()
        vae_model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=vae_model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            epochs=epochs,
            max_grad_norm=max_grad_norm,
            noise_generator=noise_generator,
        )

    # ─── Training loop ─────────────────────────────
    start_time = time.time()
    loss_train, error_observed_train, error_missing_train = [], [], []
    loss_val = []

    # ─── Setting for early stopping ─────────────────────────────
    best_val_loss = float('inf')
    patience = 6
    n_iter_validation = 50
    n_iter_min = 100
    counter = 0

    for epoch in range(epochs):
        global_epoch = epoch + start_epoch
        avg_loss = avg_KL_s = avg_KL_z = 0.0
        avg_loss_val = avg_KL_s_val = avg_KL_z_val = 0.0
        samples_list, p_params_list, q_params_list, log_p_x_total, log_p_x_missing_total = [], [], [], [], []
        tau = max(1.0 - 0.01 * global_epoch, 1e-3)

        n_batches_seen = 0
        for batch_data_list, batch_miss_list in train_loader:
            # Mask unknown data (set unobserved values to zero)
            data_list_observed = [data * batch_miss_list[:, i].view(data.shape[0], 1) for i, data in enumerate(batch_data_list)]

            # Compute loss
            optimizer.zero_grad()
            vae_res = vae_model.forward(data_list_observed, batch_data_list, batch_miss_list, tau, n_generated_dataset=1)
            vae_res["neg_ELBO_loss"].backward()
            optimizer.step()

            avg_loss += vae_res["neg_ELBO_loss"].item()
            avg_KL_s += torch.mean(vae_res["KL_s"]).item() 
            avg_KL_z += torch.mean(vae_res["KL_z"]).item()
            n_batches_seen += 1

            # Save the generated samlpes and estimated parameters !
            samples_list.append(vae_res["samples"])
            p_params_list.append(vae_res["p_params"])
            q_params_list.append(vae_res["q_params"])
            log_p_x_total.append(vae_res["log_p_x"])
            log_p_x_missing_total.append(vae_res["log_p_x_missing"])

        if n_batches_seen > 0:
            avg_loss /= n_batches_seen
            avg_KL_s /= n_batches_seen
            avg_KL_z /= n_batches_seen
        loss_train.append(avg_loss)

        # Concatenate samples in arrays
        s_total, z_total, y_total, est_data_train = statistic.samples_concatenation(samples_list)
        n_train_samples = min(est_data_train[0].shape[0], data_train.shape[0])
        # Transform discrete variables back to the original values
        data_train_transformed = data_processing.discrete_variables_transformation(data_train[:n_train_samples], feat_types_dict)
        est_data_train_transformed = data_processing.discrete_variables_transformation(est_data_train[0][:n_train_samples], feat_types_dict)
        error_observed_samples, error_missing_samples = statistic.error_computation(data_train_transformed, est_data_train_transformed,
                                                                                    feat_types_dict,
                                                                                    miss_mask_train[:n_train_samples])
        # Save error
        error_observed_train.append(torch.mean(error_observed_samples))
        error_missing_train.append(torch.mean(error_missing_samples))

        if verbose and global_epoch % 100 == 0:
            visualization.print_loss(global_epoch, start_time, -avg_loss, avg_KL_s, avg_KL_z)

        if epoch % n_iter_validation == 0:
            with torch.no_grad():            
                for i in range(n_batches_test):
                    data_list_test, miss_list_test = data_processing.next_batch(data_test, feat_types_dict, miss_mask_test, batch_test_size, i)
                    data_list_observed_test = [data * miss_list_test[:, i].view(batch_test_size, 1) for i, data in enumerate(data_list_test)]
                    vae_res_test = vae_model.forward(data_list_observed_test, data_list_test, miss_list_test, tau=1e-3, n_generated_dataset=1)
                    avg_loss_val += vae_res_test["neg_ELBO_loss"].item() / max(n_batches_test, 1)
                    avg_KL_s_val += torch.mean(vae_res_test["KL_s"]).item() / max(n_batches_test, 1)
                    avg_KL_z_val += torch.mean(vae_res_test["KL_z"]).item() / max(n_batches_test, 1)
            loss_val.append(avg_loss_val)
            if avg_loss_val >= best_val_loss:
                counter += 1
            else: 
                best_val_loss = avg_loss_val
                counter = 0
            if counter >= patience and epoch >= n_iter_min:
                print(f"Early stopping at epoch {global_epoch}.")
                early_stop = True
                break
        else:
            loss_val.append(torch.nan)

    if differential_privacy:
        # ─── Report consumed privacy budget ────────────────────────────────
        final_epsilon = privacy_engine.get_epsilon(delta=target_delta)
        if verbose:
            print(f"Training finished. Final ε: {final_epsilon:.2f} (δ={target_delta})")
            if early_stop:
                print(f"Early stop: actual ε = {final_epsilon:.2f} < requested ε ≤ {target_epsilon}")
            else:
                print(f"Full training: actual ε = {final_epsilon:.2f}, requested ε ≤ {target_epsilon}")
        unwrapped_model = vae_model._module
        unwrapped_model.eval()
        # return unwrapped_model, loss_train, loss_val, final_epsilon
        return unwrapped_model, loss_train, loss_val
    else:
        if verbose:
            print("Training finished.")
        vae_model.eval()
        return vae_model, loss_train, loss_val



# ───────────────────────────────────────────────────────────
# GENERATION FUNCTIONS: with conditioning on a feature value.
# ───────────────────────────────────────────────────────────

def generate_from_condition_HIVAE(vae_model, df, miss_mask, true_miss_mask, feat_types_dict, n_generated_dataset, n_generated_sample=None, from_prior=False, condition={'var': "treatment", 'value': 0.0, 'n_samples': 300}):
    """
        Generation from a trained HIVAE, with optional conditioning on a feature value. 
        If `condition` is not None, generation is repeated until `condition['n_samples']` samples matching the condition are obtained for each generated dataset. 
        If `n_generated_sample` is specified, generation is performed on an extended dataset of that size (with samples drawn with replacement from the original data) 
        to increase the pool of candidates for conditioning; otherwise, generation is performed on the original data. 
        If `from_prior=True`, samples are generated from the prior instead of the posterior.
    """
    data = torch.from_numpy(df.values)
    features = df.columns.tolist()
    cond_feature_idx = features.index(condition['var'])
    miss_mask = torch.multiply(miss_mask, true_miss_mask)

    if n_generated_sample is None:
        n_generated_sample = data.shape[0]
        data_ext = data
        miss_mask_ext = miss_mask
    else:
        indices = torch.cat((torch.arange(0, data.shape[0]), torch.randint(0, data.shape[0], (n_generated_sample - data.shape[0],))))
        data_ext = data[indices]
        miss_mask_ext = miss_mask[indices]

    batch_size = n_generated_sample

    with torch.no_grad():
        min_shape = 0
        est_data_gen_transformed = []
        i = 0
        while min_shape < condition['n_samples']:
            if i > 0:
                est_data_gen_transformed = [t[:, :min_shape, :] for t in est_data_gen_transformed]

            samples_list = []
            data_list, miss_list = data_processing.next_batch(data_ext, feat_types_dict, miss_mask_ext, batch_size, 0)
            # Mask unknown data (set unobserved values to zero)
            data_list_observed = [data * miss_list[:, i].view(batch_size, 1) for i, data in enumerate(data_list)]

            if from_prior:
                _, normalization_params = data_processing.normalize_features(
                    data_list_observed, vae_model.feat_types_list, miss_list,
                    feat_normalization_globals=vae_model.feat_normalization_globals,
                )
                s_samples = torch.randint(0, vae_model.s_dim, (n_generated_sample,))
                samples_s = torch.nn.functional.one_hot(s_samples, num_classes=vae_model.s_dim).float()
                mean_pz, log_var_pz = statistic.z_prior_GMM(samples_s, vae_model.z_distribution_layer)
                eps = torch.randn_like(mean_pz)
                samples_z = mean_pz + torch.exp(log_var_pz / 2) * eps  # mean_pz + eps
                samples_y = vae_model.y_layer(samples_z)
                grouped_samples_y = data_processing.y_partition(samples_y, vae_model.feat_types_list, vae_model.y_dim_partition)

                # Compute θ parameters
                theta = theta_estimation.theta_estimation_from_ys(grouped_samples_y, samples_s, vae_model.feat_types_list, miss_list, vae_model.theta_layer)

                # Compute log-likelihood and reconstructed data
                _, _, _, samples_x = likelihood.loglik_evaluation(data_list, vae_model.feat_types_list, miss_list, theta, normalization_params, n_generated_dataset)
                samples = {"s": samples_s, "z": samples_z, "y": samples_y, "x": samples_x}
                samples_list.append(samples)

            else:
                vae_res = vae_model.forward(data_list_observed, data_list, miss_list, tau=1e-3, n_generated_dataset=n_generated_dataset)
                samples_list.append(vae_res["samples"])

            #Concatenate samples in arrays
            est_data_gen = statistic.samples_concatenation(samples_list)[-1]
            for j in range(n_generated_dataset):
                est_data = est_data_gen[j][est_data_gen[j][:, cond_feature_idx] == condition["value"]]
                data_trans = data_processing.discrete_variables_transformation(est_data, feat_types_dict)
                data_trans = data_processing.survival_variables_transformation(data_trans, feat_types_dict)
                if i == 0:
                    est_data_gen_transformed.append(data_trans.unsqueeze(0))
                else:
                    est_data_gen_transformed[j] = torch.cat((est_data_gen_transformed[j], data_trans.unsqueeze(0)), dim=1)

            shapes = [t.shape[1] for t in est_data_gen_transformed]
            min_shape = min(shapes)
            i += 1

        est_data_gen_transformed = [t[:, :condition['n_samples'], :] for t in est_data_gen_transformed]
        est_data_gen_transformed = torch.cat(est_data_gen_transformed, dim=0)

        return est_data_gen_transformed


# ──────────────────────────────────────────────────────────────
# GENERATION FUNCTIONS: without conditioning on a feature value.
# ──────────────────────────────────────────────────────────────
 
def generate_from_HIVAE(vae_model, data, miss_mask, true_miss_mask, feat_types_dict, n_generated_dataset, n_generated_sample=None, from_prior=False):
    """
        Generation from a trained HIVAE, without conditioning. 
        If `n_generated_sample` is specified, generation is performed on an extended dataset of that size (with samples drawn with replacement from the original data) 
        to increase the pool of candidates for conditioning; otherwise, generation is performed on the original data. 
        If `from_prior=True`, samples are generated from the prior instead of the posterior.
    """
    miss_mask = torch.multiply(miss_mask, true_miss_mask)
    if n_generated_sample is None:
        n_generated_sample = data.shape[0]
        data_ext = data
        miss_mask_ext = miss_mask
    else:
        indices = torch.cat((torch.arange(0, data.shape[0]), torch.randint(0, data.shape[0], (n_generated_sample - data.shape[0],))))
        data_ext = data[indices]
        miss_mask_ext = miss_mask[indices]

    batch_size = n_generated_sample
 
    with torch.no_grad():
        samples_list = []
        data_list, miss_list = data_processing.next_batch(data_ext, feat_types_dict, miss_mask_ext, batch_size, 0)
        # Mask unknown data (set unobserved values to zero)
        data_list_observed = [data * miss_list[:, i].view(batch_size, 1) for i, data in enumerate(data_list)]

        if from_prior:
            _, normalization_params = data_processing.normalize_features(
                data_list_observed, feat_types_dict, miss_list,
                feat_normalization_globals=vae_model.feat_normalization_globals,
            )
            s_samples = torch.randint(0, vae_model.s_dim, (n_generated_sample,))
            samples_s = torch.nn.functional.one_hot(s_samples, num_classes=vae_model.s_dim).float()
            mean_pz, log_var_pz = statistic.z_prior_GMM(samples_s, vae_model.z_distribution_layer)
            eps = torch.randn_like(mean_pz)
            samples_z = mean_pz + torch.exp(log_var_pz / 2) * eps  # mean_pz + eps
            samples_y = vae_model.y_layer(samples_z)
            grouped_samples_y = data_processing.y_partition(samples_y, feat_types_dict, vae_model.y_dim_partition)

            # Compute θ parameters    
            theta = theta_estimation.theta_estimation_from_ys(grouped_samples_y, samples_s, feat_types_dict, miss_list, vae_model.theta_layer)

            # Compute log-likelihood and reconstructed data
            _, _, _, samples_x = likelihood.loglik_evaluation(data_list, feat_types_dict, miss_list, theta, normalization_params, n_generated_dataset)
            samples = {"s": samples_s, "z": samples_z, "y": samples_y, "x": samples_x}
            samples_list.append(samples)

        else:
            vae_res = vae_model.forward(data_list_observed, data_list, miss_list, tau=1e-3, n_generated_dataset=n_generated_dataset)
            samples_list.append(vae_res["samples"])
        
        # Concatenate samples in arrays
        est_data_gen = statistic.samples_concatenation(samples_list)[-1]
        est_data_gen_transformed = []
        for j in range(n_generated_dataset):
            data_trans = data_processing.discrete_variables_transformation(est_data_gen[j], feat_types_dict)
            data_trans = data_processing.survival_variables_transformation(data_trans, feat_types_dict)
            est_data_gen_transformed.append(data_trans.unsqueeze(0))
            
        est_data_gen_transformed = torch.cat(est_data_gen_transformed, dim=0)

        return est_data_gen_transformed


# ──────────────────────────────────────────────────────────────────────────────────────────────
# COMPLETE RUNNING FUNCTIONS: train + generate + optional plotting, with hyperparameter handling.
# ──────────────────────────────────────────────────────────────────────────────────────────────

# Hyperparameter defaults shared by `run` and `run_alt`. Piecewise-only keys (`n_intervals`, `n_layers_surv_piecewise`) are deliberately NOT in `_DEFAULT_HP`. 
# They remain valid kwargs (listed in `_VALID_HP_KEYS`) so a user who wants piecewise can opt in explicitly.
_DEFAULT_HP = {
    "lr": 1e-3, "batch_size": 100, "z_dim": 20, "y_dim": 15, "s_dim": 20,
    "max_grad_norm": 1.0, "epochs": 1000,
}
_PIECEWISE_HP_KEYS = {"n_intervals", "n_layers_surv_piecewise"}
_VALID_HP_KEYS = set(_DEFAULT_HP) | _PIECEWISE_HP_KEYS

def _resolve_params(params, hp_overrides):
    """
        Merge `_DEFAULT_HP` with the user's partial `params` dict and any
        individual HP kwargs (`hp_overrides`). Precedence (low → high):
        defaults → params → hp_overrides.

        Raises TypeError if `hp_overrides` contains a name that isn't a known
        HP, so typos like `learning_rate=1e-4` fail loudly instead of being
        silently merged and ignored downstream.
    """
    unknown = set(hp_overrides) - _VALID_HP_KEYS
    if unknown:
        raise TypeError(
            f"Unexpected HP kwargs: {sorted(unknown)}. "
            f"Valid HP names: {sorted(_VALID_HP_KEYS)}"
        )
    return {**_DEFAULT_HP, **(params or {}), **hp_overrides}

def _validate_norm_mode(norm_mode, differential_privacy):
    """
        Validate the `norm_mode` switch and its compatibility with DP.

        `norm_mode` selects how the encoder's input normalization statistics
        are obtained (see data_processing.resolve_feat_normalization_globals):
        - "global" : frozen training-set statistics for every feature.
        - "batch"  : per-batch statistics for every feature (mean/var for
                    real / pos / legacy 'surv', min/max for the survival
                    families).

        Per-batch statistics couple samples inside the forward pass, which
        breaks Opacus per-sample gradients and therefore the DP guarantee, so
        norm_mode="batch" is rejected whenever differential_privacy=True.
    """
    if norm_mode not in ("global", "batch"):
        raise ValueError(f"norm_mode must be 'global' or 'batch', got {norm_mode!r}.")
    if differential_privacy and norm_mode == "batch":
        raise ValueError(
            "norm_mode='batch' is incompatible with differential_privacy=True: "
            "per-batch statistics couple samples inside the forward pass and "
            "break Opacus per-sample gradients / the DP guarantee. "
            "Use norm_mode='global'."
        )

# ─── MedianPruner screening helpers (shared by the optuna_* searches) ───
# Pruning trains each trial for `screening_epochs` first, reports the model's validation ELBO loss at that point,
# and lets Optuna's MedianPruner stop trials that look worse than the running median before the remaining epochs are spent. 
# `n_trials` therefore stays the *total* number of trials Optuna runs (study.optimize(n_trials=n_trials)) 
# — pruned trials simply finish early at the screening epoch. 
def _last_finite_val_loss(loss_val):
    """
        Return the last non-NaN entry of a train_HIVAE* `loss_val` list as a
        float. Validation loss is only recorded every n_iter_validation epochs
        (other epochs hold NaN), so the final list element is not necessarily a
        real value — pick the last finite one. Falls back to +inf when the list
        has no finite entry (e.g. screening shorter than one validation cycle),
        which makes such a trial look maximally prunable rather than crashing.
    """
    finite = [v for v in loss_val if v == v]  # v == v is False for NaN
    return float(finite[-1]) if finite else float("inf")

def _report_and_maybe_prune(trial, screening_val_losses, screening_epochs):
    """
        Report the mean screening validation loss to `trial` at
        step=screening_epochs and raise optuna.TrialPruned() if the pruner
        decides this trial is not worth continuing. `screening_val_losses` is
        one value per trained model (a single-element list outside the
        multi-model searches).
    """
    finite = [v for v in screening_val_losses if v == v]
    avg_val = float(np.mean(finite)) if finite else float("inf")
    print(f"Screening avg val loss @ epoch {screening_epochs}: {avg_val}")
    trial.report(avg_val, step=screening_epochs)
    if trial.should_prune():
        raise optuna.TrialPruned()

def _make_pruner(generator_name, n_startup_trials):
    """
        MedianPruner for non-DP generators, NopPruner for DP ones (the caller
        also skips the screening split in the DP case, so NopPruner is just a
        belt-and-braces no-op). Attached to the study at create/load time.
    """
    if "_DP" in generator_name:
        return optuna.pruners.NopPruner()
    return optuna.pruners.MedianPruner(n_startup_trials=n_startup_trials)


def run(df, miss_mask, true_miss_mask, feat_types_dict,  n_generated_dataset, n_generated_sample=None,
        params=None,
        verbose=True, plot=False, gen_from_prior=False, condition=None, differential_privacy=False,
        norm_mode="global",
        seed=1,
        target_epsilon=None, target_delta=1e-5,
        **hp_overrides):
    """
        End-to-end entry point: build a HIVAE, train it on `df`, and generate `n_generated_dataset` synthetic datasets.
        `condition` (optional) routes generation through generate_from_condition_HIVAE; `gen_from_prior=True` samples z directly from the prior. 
        With `plot=True`, the training/val loss curves are displayed at the end.
        Hyperparameters can be supplied three ways (combined via `_resolve_params`):
            1. `params=<full or partial dict>` — typically HPO output.
            2. Individual kwargs, e.g. `run(..., lr=1e-4, epochs=500)`.
            3. Neither — `_DEFAULT_HP` is used.
            Piecewise mode is opted into by including `n_intervals` (and `n_layers_surv_piecewise`) in `params` or the kwargs.
    """
    params = _resolve_params(params, hp_overrides)
    _validate_norm_mode(norm_mode, differential_privacy)

    set_seed(seed=seed)
    model_name = "HIVAE_inputDropout" # "HIVAE_factorized"

    miss_mask = miss_mask
    true_miss_mask = true_miss_mask
    dim_latent_z = params["z_dim"]
    dim_latent_y = params["y_dim"]
    dim_latent_s = params["s_dim"]
    epochs = params["epochs"]
    lr = params["lr"]
    batch_size = params["batch_size"]
    batch_size = min(batch_size, int(0.9*df.shape[0])) # Adjust batch size if larger than dataset
    if "n_intervals" in params:
        # HI_VAE piecewise
        intervals = get_intervals(df, params["n_intervals"])
        n_layers = params["n_layers_surv_piecewise"]
    else:
        intervals = None
        n_layers = None 

    # Create PyTorch HVAE model
    model_loading = getattr(importlib.import_module("src"), model_name)
    model_hivae = model_loading(input_dim=df.shape[1],
                            z_dim=dim_latent_z,
                            y_dim=dim_latent_y,
                            s_dim=dim_latent_s, 
                            y_dim_partition=None,
                            feat_types_dict=feat_types_dict,
                            intervals_surv_piecewise=intervals,
                            n_layers_surv_piecewise=n_layers
                            )
    data = torch.from_numpy(df.values)
    if differential_privacy == True and target_epsilon is None:
        raise ValueError("target_epsilon must be set when differential_privacy=True.")
    model_hivae, loss_train, loss_val = train_HIVAE(model_hivae, data, miss_mask, true_miss_mask, feat_types_dict, 
                                                    batch_size, lr, epochs, verbose, seed=seed, norm_mode=norm_mode, 
                                                    differential_privacy=differential_privacy, target_epsilon=target_epsilon, target_delta=target_delta, max_grad_norm=params["max_grad_norm"])
    
    if isinstance(n_generated_sample, list):
        est_data_gen_transformed_list = []
        for n_generated_sample_ in n_generated_sample:
            if condition is not None:
                est_data_gen_transformed = generate_from_condition_HIVAE(model_hivae, df, miss_mask, true_miss_mask, feat_types_dict, n_generated_dataset, n_generated_sample_, from_prior=gen_from_prior, condition=condition)
            else:
                est_data_gen_transformed = generate_from_HIVAE(model_hivae, data, miss_mask, true_miss_mask, feat_types_dict, n_generated_dataset, n_generated_sample_, from_prior=gen_from_prior)
            est_data_gen_transformed_list.append(est_data_gen_transformed)

        return est_data_gen_transformed_list
    else:
        if condition is not None:
            est_data_gen_transformed = generate_from_condition_HIVAE(model_hivae, df, miss_mask, true_miss_mask, feat_types_dict, n_generated_dataset, n_generated_sample, from_prior=gen_from_prior, condition=condition)
        else:
            est_data_gen_transformed = generate_from_HIVAE(model_hivae, data, miss_mask, true_miss_mask, feat_types_dict, n_generated_dataset, n_generated_sample, from_prior=gen_from_prior)

        if plot:
            loss_track = {"epoch": list(range(1, len(loss_train) + 1)),
                        "loss_train": loss_train,
                        "loss_val": loss_val}

            loss_df = pd.DataFrame(loss_track)
            loss_df_melted = loss_df.melt(id_vars="epoch", value_vars=["loss_train", "loss_val"],
                                        var_name="Loss Type", value_name="Loss")

            # Plot
            plt.figure(figsize=(10, 5))
            sns.lineplot(data=loss_df_melted, x="epoch", y="Loss", hue="Loss Type")
            plt.title("Loss evolution", fontweight="bold")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend(title="Loss Type")
            plt.tight_layout()
            plt.show()
        # if differential_privacy:
        #     return est_data_gen_transformed, final_eps
        # else:
        return est_data_gen_transformed
    


# ──────────────────────────────────────────────────────────────
# HYPERPARAMETER OPTIMIZATION 
# ──────────────────────────────────────────────────────────────

from synthcity.plugins.core.distribution import (
    CategoricalDistribution,
    Distribution,
    FloatDistribution,
    IntegerDistribution,
)
import optuna
from synthcity.utils.optuna_sample import suggest_all
from sklearn.model_selection import KFold
from synthcity.utils.reproducibility import clear_cache, enable_reproducible_results
from synthcity.metrics.eval import Metrics
from synthcity.plugins.core.dataloader import SurvivalAnalysisDataLoader


def hyperparameter_space(data, n_splits, generator_name, tune_params=None):
    """
        Define the hyperparameter space for the model

        Parameters to optimize: z_dim, y_dim, s_dim, batch_size, lr, n_layers_surv_piecewise
        (and max_grad_norm when generator_name contains "_DP").

        If `tune_params` is None (default), all available hyperparameters are tuned.
        Otherwise, only those whose name appears in `tune_params` are kept.
    """
    n_samples = data.shape[0]
    hp_space = [
        CategoricalDistribution(name="lr", choices=[1e-4, 2e-4, 1e-3, 2e-3, 3e-3, 5e-3]),
        CategoricalDistribution(name="batch_size", choices=get_batchsize(n_samples, n_splits) + [32]),
        IntegerDistribution(name="z_dim", low=10, high=200, step=10),
        IntegerDistribution(name="y_dim", low=10, high=200, step=10),
        IntegerDistribution(name="s_dim", low=10, high=200, step=10),
    ]
    if "HI-VAE_piecewise" in generator_name:
       hp_space.append(CategoricalDistribution(name="n_layers_surv_piecewise", choices=[1, 2]))
       hp_space.append(CategoricalDistribution(name="n_intervals", choices=[5, 10, 15, 20]))

    if "_DP" in generator_name:
        hp_space.append(CategoricalDistribution(name="max_grad_norm", choices=[0.1, 0.3, 1.0, 3.0, 10.0]))
        hp_space.append(CategoricalDistribution(name="epochs", choices=[20, 50, 100, 200, 400, 1000]))

    if tune_params is not None:
        hp_space = [d for d in hp_space if d.name in tune_params]

    return hp_space

def get_n_hyperparameters(generator_name):
    """
        Returns the number of hyperparameters for the SurVAE model.
    """
    hp_space = hyperparameter_space(data=np.zeros(10), n_splits=5, generator_name=generator_name)  # Dummy data for space definition
    return len(hp_space)

def get_intervals(data, n_intervals):
    """
        Intervals
    """
    T_surv = torch.Tensor(data.time)
    T_surv_norm = (T_surv - T_surv.min()) / (T_surv.max() - T_surv.min())
    T_intervals = torch.linspace(0., T_surv_norm.max(), n_intervals)
    T_intervals = torch.cat([T_intervals, torch.tensor([2 * T_intervals[-1] - T_intervals[-2]])])
    intervals = [(T_intervals[i].item(), T_intervals[i + 1].item()) for i in range(len(T_intervals) - 1)]
    return intervals

def get_batchsize(n_samples, n_splits):
    """
        Batch size
    """
    batch_size_ratio = [.15, .2, .25, .4, .6, .75]
    batch_size = [int(ratio * n_samples * (n_splits - 1) / n_splits) for ratio in batch_size_ratio]

    return batch_size

def _validate_tune_and_fixed_params(data, n_splits, generator_name, tune_params, fixed_params):
    """
        Fail fast (before any Optuna trial runs) if hyperparameters needed by the
        objective won't be available at trial time.

        - `epochs` is required by every objective. For DP runs it's in the HP
        space (tuned by default, or — if excluded from `tune_params` — must be
        in `fixed_params`). For non-DP runs it isn't in the HP space, so it
        must always be supplied via `fixed_params`.
        - When `tune_params` is given, every other HP in the space that isn't
        tuned must also be in `fixed_params`.
    """
    full_hp_space = hyperparameter_space(data, n_splits, generator_name)
    available = {d.name for d in full_hp_space}
    fixed_keys = set(fixed_params or {})

    if "epochs" not in available and "epochs" not in fixed_keys:
        raise ValueError(
            f"`epochs` is required by the HPO objective but is not in the HP "
            f"space for generator_name={generator_name!r} (only DP HP spaces "
            f"include it). Pass it via fixed_params, e.g. "
            f"fixed_params={{'epochs': 1000}}."
        )
    if tune_params is None:
        return
    unknown = set(tune_params) - available
    if unknown:
        raise ValueError(
            f"tune_params contains names not in the HP space for "
            f"generator_name={generator_name!r}: {sorted(unknown)}. "
            f"Available: {sorted(available)}"
        )
    missing = available - set(tune_params) - fixed_keys
    if missing:
        raise ValueError(
            f"Hyperparameters {sorted(missing)} are neither tuned nor "
            f"provided in fixed_params. Add them to tune_params or supply "
            f"values in fixed_params."
        )


def optuna_hyperparameter_search(df, miss_mask, true_miss_mask, feat_types_dict, n_generated_dataset, n_splits, n_trials, 
                                 columns, generator_name, n_generated_sample = None, study_name='optuna_study_surv_hivae', 
                                 metric='survival_km_distance', method='', gen_from_prior=False, condition=None, cond_df=None, seed=10,
                                 target_epsilon=None, target_delta=1e-5,
                                 tune_params=None, fixed_params=None, norm_mode="global",
                                 screening_epochs=800, n_startup_trials=20, do_prune=False):
    
    differential_privacy = "_DP" in generator_name
    if differential_privacy and target_epsilon is None:
        raise ValueError("target_epsilon must be set when generator_name contains '_DP'.")

    _validate_norm_mode(norm_mode, differential_privacy=differential_privacy)
    _validate_tune_and_fixed_params(df, n_splits, generator_name, tune_params, fixed_params)

    model_name = "HIVAE_inputDropout" # "HIVAE_factorized"
    miss_mask = miss_mask
    true_miss_mask = true_miss_mask
    if condition is not None and cond_df is not None:
        cond_full_data_loader =  SurvivalAnalysisDataLoader(cond_df, target_column = "censor", time_to_event_column = "time")
 
    def objective(trial: optuna.Trial):
        set_seed(seed=seed)
        hp_space = hyperparameter_space(df, n_splits, generator_name, tune_params=tune_params)
        sampled = suggest_all(trial, hp_space) # dict of tuned hyperparameters
        params = {**(fixed_params or {}), **sampled}
        epochs = params["epochs"]
        max_grad_norm = params.get("max_grad_norm", None)
        if "HI-VAE_piecewise" in generator_name:
            intervals = get_intervals(df, params["n_intervals"])
            n_layers = params["n_layers_surv_piecewise"]
        else:
            intervals = None
            n_layers = None
        print(f"trial_{trial.number}")
        print(f"Hyperparameters: {params}")
        model_loading = getattr(importlib.import_module("src"), model_name)
        data = torch.from_numpy(df.values)

        def _screen_train(model, data_, miss_, true_miss_, batch_size, differential_privacy_, target_epsilon_, target_delta_, max_grad_norm_, split_seed=seed):
            # Train `model` with optional MedianPruner screening. When pruning is active and screening_epochs < epochs, train a screening chunk, 
            # report its validation loss / maybe prune, then train the remaining epochs from the screening weights.
            # numpy is reseeded before each chunk so the two chunks share train_HIVAE's internal train/val split.
            if do_prune and screening_epochs < epochs:
                np.random.seed(split_seed)
                model, _, lv = train_HIVAE(model, data_, miss_, true_miss_, feat_types_dict, 
                                           batch_size, params["lr"], screening_epochs, norm_mode=norm_mode, 
                                            differential_privacy=differential_privacy_, target_epsilon=target_epsilon_, 
                                            target_delta=target_delta_, max_grad_norm=max_grad_norm_)
                _report_and_maybe_prune(trial, [_last_finite_val_loss(lv)], screening_epochs)
                np.random.seed(split_seed)
                model, _, _ = train_HIVAE(model, data_, miss_, true_miss_, feat_types_dict,
                                            batch_size, params["lr"], epochs - screening_epochs,
                                            norm_mode=norm_mode, start_epoch=screening_epochs,
                                            differential_privacy=differential_privacy_, target_epsilon=target_epsilon_, 
                                            target_delta=target_delta_, max_grad_norm=max_grad_norm_)
            else:
                model, _, _ = train_HIVAE(model, data_, miss_, true_miss_, feat_types_dict,
                                            batch_size, params["lr"], epochs, norm_mode=norm_mode, 
                                            differential_privacy=differential_privacy_, target_epsilon=target_epsilon_, 
                                            target_delta=target_delta_, max_grad_norm=max_grad_norm_)
            return model

        scores = []
        try:
            if method == 'train_full_gen_full':

                full_data_loader = SurvivalAnalysisDataLoader(df, target_column = "censor", time_to_event_column = "time")
                # Train
                batch_size = params["batch_size"]
                batch_size = min(batch_size, int(0.9*data.shape[0]))
                model_hivae = model_loading(input_dim=data.shape[1],
                            z_dim=params["z_dim"],
                            y_dim=params["y_dim"],
                            s_dim=params["s_dim"],
                            y_dim_partition=None,
                            feat_types_dict=feat_types_dict,
                            intervals_surv_piecewise=intervals,
                            n_layers_surv_piecewise=n_layers)

                model_hivae = _screen_train(model_hivae, data, miss_mask, true_miss_mask, batch_size, differential_privacy, target_epsilon, target_delta, max_grad_norm)
                # Generate
                if condition is not None:
                    est_data_gen_transformed = generate_from_condition_HIVAE(model_hivae, df, miss_mask, true_miss_mask,
                                                                            feat_types_dict, n_generated_dataset, n_generated_sample=data.shape[0], from_prior=gen_from_prior, condition=condition)

                    tensor_list = list(est_data_gen_transformed)
                    full_data_tensor = torch.cat(tensor_list, dim=0)
                    df_gen_data = pd.DataFrame(full_data_tensor.numpy(), columns=columns)
                    df_gen_data["time"] = df_gen_data["time"].clip(lower=1e-6)
                    gen_data = SurvivalAnalysisDataLoader(df_gen_data, target_column="censor", time_to_event_column="time")
                    clear_cache()
                    evaluation = Metrics().evaluate(X_gt=cond_full_data_loader, # can be dataloaders or dataframes
                                                    X_syn=gen_data, 
                                                    reduction='mean', # default mean
                                                    n_histogram_bins=10, # default 10
                                                    n_folds=1,
                                                    metrics={'stats': ['survival_km_distance']},
                                                    task_type='survival_analysis', 
                                                    use_cache=True)
                else:
                    n_gen_sample = n_generated_sample if n_generated_sample is not None else data.shape[0]
                    est_data_gen_transformed = generate_from_HIVAE(model_hivae, data, miss_mask, true_miss_mask,
                                                                feat_types_dict, n_generated_dataset=n_generated_dataset, n_generated_sample=n_gen_sample, from_prior=gen_from_prior)
                
                    tensor_list = list(est_data_gen_transformed)
                    full_data_tensor = torch.cat(tensor_list, dim=0)
                    df_gen_data = pd.DataFrame(full_data_tensor.numpy(), columns=columns)
                    df_gen_data["time"] = df_gen_data["time"].clip(lower=1e-6)
                    gen_data = SurvivalAnalysisDataLoader(df_gen_data, target_column="censor", time_to_event_column="time")
                    clear_cache()
                    evaluation = Metrics().evaluate(X_gt=full_data_loader, # can be dataloaders or dataframes
                                                    X_syn=gen_data, 
                                                    reduction='mean', # default mean
                                                    n_histogram_bins=10, # default 10
                                                    n_folds=1,
                                                    metrics={'stats': ['survival_km_distance']},
                                                    task_type='survival_analysis', 
                                                    use_cache=True)
                scores = evaluation.T[["stats.survival_km_distance.abs_optimism"]].T["mean"].values[0]

            elif method == 'train_train_gen_full':
                # Train-test split on control
                train_test_share = .8
                n_samples = data.shape[0]
                n_train_samples = int(train_test_share * n_samples)
                train_index = np.random.choice(n_samples, n_train_samples, replace=False)
                test_index = [i for i in np.arange(n_samples) if i not in train_index]

                train_data, test_data = data[train_index], data[test_index]
                train_miss_mask = miss_mask[train_index]
                train_true_miss_mask = true_miss_mask[train_index]

                full_data_loader = SurvivalAnalysisDataLoader(df, target_column = "censor", time_to_event_column = "time")

                # Train
                batch_size = params["batch_size"]
                batch_size = min(batch_size, train_data.shape[0])
                model_hivae = model_loading(input_dim=data.shape[1],
                            z_dim=params["z_dim"],
                            y_dim=params["y_dim"],
                            s_dim=params["s_dim"],
                            y_dim_partition=None,
                            feat_types_dict=feat_types_dict,
                            intervals_surv_piecewise=intervals,
                            n_layers_surv_piecewise=n_layers)
                model_hivae = _screen_train(model_hivae, data, miss_mask, true_miss_mask, batch_size, differential_privacy, target_epsilon, target_delta, max_grad_norm)
                # Generate
                if condition is not None:
                    est_data_gen_transformed = generate_from_condition_HIVAE(model_hivae, df, miss_mask, true_miss_mask,
                                                                            feat_types_dict, n_generated_dataset, n_generated_sample=data.shape[0], from_prior=gen_from_prior, condition=condition)
                    tensor_list = list(est_data_gen_transformed)
                    full_data_tensor = torch.cat(tensor_list, dim=0)
                    df_gen_data = pd.DataFrame(full_data_tensor.numpy(), columns=columns)
                    df_gen_data["time"] = df_gen_data["time"].clip(lower=1e-6)
                    gen_data = SurvivalAnalysisDataLoader(df_gen_data, target_column="censor", time_to_event_column="time")
                    clear_cache()
                    evaluation = Metrics().evaluate(X_gt=cond_full_data_loader, # can be dataloaders or dataframes
                                                    X_syn=gen_data, 
                                                    reduction='mean', # default mean
                                                    n_histogram_bins=10, # default 10
                                                    n_folds=1,
                                                    metrics={'stats': ['survival_km_distance']},
                                                    task_type='survival_analysis',
                                                    use_cache=True)
                else:
                    n_gen_sample = n_generated_sample if n_generated_sample is not None else data.shape[0]
                    est_data_gen_transformed = generate_from_HIVAE(model_hivae, data, miss_mask, true_miss_mask,
                                                                feat_types_dict, n_generated_dataset=n_generated_dataset, n_generated_sample=data.shape[0], from_prior=gen_from_prior)
                    tensor_list = list(est_data_gen_transformed)
                    full_data_tensor = torch.cat(tensor_list, dim=0)
                    df_gen_data = pd.DataFrame(full_data_tensor.numpy(), columns=columns)
                    df_gen_data["time"] = df_gen_data["time"].clip(lower=1e-6)
                    gen_data = SurvivalAnalysisDataLoader(df_gen_data, target_column="censor", time_to_event_column="time")
                    clear_cache()
                    evaluation = Metrics().evaluate(X_gt=full_data_loader, # can be dataloaders or dataframes
                                                    X_syn=gen_data, 
                                                    reduction='mean', # default mean
                                                    n_histogram_bins=10, # default 10
                                                    n_folds=1,
                                                    metrics={'stats': ['survival_km_distance']},
                                                    task_type='survival_analysis', 
                                                    use_cache=True)
                scores = evaluation.T[["stats.survival_km_distance.abs_optimism"]].T["mean"].values[0]

            elif method == 'train_train_gen_test':
                # Train-test split on control
                train_test_share = .8
                n_samples = data.shape[0]
                n_train_samples = int(train_test_share * n_samples)
                train_index = np.random.choice(n_samples, n_train_samples, replace=False)
                test_index = [i for i in np.arange(n_samples) if i not in train_index]

                train_data, test_data = data[train_index], data[test_index]
                df_test_data = df.iloc[test_index]
                test_data_loader = SurvivalAnalysisDataLoader(df_test_data, target_column = "censor", time_to_event_column = "time")
                train_miss_mask, test_miss_mask = miss_mask[train_index], miss_mask[test_index]
                train_true_miss_mask, test_true_miss_mask = true_miss_mask[train_index], true_miss_mask[test_index]

                # Train
                batch_size = params["batch_size"]
                batch_size = min(batch_size, train_data.shape[0])
                model_hivae = model_loading(input_dim=data.shape[1],
                            z_dim=params["z_dim"],
                            y_dim=params["y_dim"],
                            s_dim=params["s_dim"],
                            y_dim_partition=None,
                            feat_types_dict=feat_types_dict,
                            intervals_surv_piecewise=intervals,
                            n_layers_surv_piecewise=n_layers)
                model_hivae = _screen_train(model_hivae, data, miss_mask, true_miss_mask, batch_size, differential_privacy, target_epsilon, target_delta, max_grad_norm)
                # Generate
                if condition is not None:
                    raise NotImplementedError("Condition not implemented for this method")
                else:
                    est_data_gen_transformed = generate_from_HIVAE(model_hivae, test_data, test_miss_mask, test_true_miss_mask,
                                                                feat_types_dict, n_generated_dataset=n_generated_dataset, n_generated_sample=test_data.shape[0], from_prior=gen_from_prior)
                    tensor_list = list(est_data_gen_transformed)
                    full_data_tensor = torch.cat(tensor_list, dim=0)
                    df_gen_data = pd.DataFrame(full_data_tensor.numpy(), columns=columns)
                    df_gen_data["time"] = df_gen_data["time"].clip(lower=1e-6)
                    gen_data = SurvivalAnalysisDataLoader(df_gen_data, target_column="censor", time_to_event_column="time")
                    clear_cache()
                    evaluation = Metrics().evaluate(X_gt=test_data_loader, # can be dataloaders or dataframes
                                                    X_syn=gen_data, 
                                                    reduction='mean', # default mean
                                                    n_histogram_bins=10, # default 10
                                                    n_folds=1,
                                                    metrics={'stats': ['survival_km_distance']},
                                                    task_type='survival_analysis', 
                                                    use_cache=True)
                scores = evaluation.T[["stats.survival_km_distance.abs_optimism"]].T["mean"].values[0]
            else:
                raise ValueError("Invalid method")
            
            print(f"Score: {np.mean(scores)}")
        except optuna.TrialPruned:
            raise
        except Exception as e:  # invalid set of params
            print(f"{type(e).__name__}: {e}")
            print(params)
            if isinstance(e, ValueError) and "invalid values" in str(e).lower():
                raise optuna.exceptions.TrialPruned()
            raise
        return np.mean(scores)


    pruner = _make_pruner(generator_name, n_startup_trials)
    db_file = study_name + '.db'
    if os.path.exists(db_file):
        print("This optuna study ({}) already exists. We load the study from the existing file.".format(db_file))
        study = optuna.load_study(study_name=study_name, storage='sqlite:///'+study_name+'.db', pruner=pruner)
    else:
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="minimize", study_name=study_name, storage='sqlite:///'+study_name+'.db', sampler=sampler, pruner=pruner)
        if "HI-VAE_piecewise" in generator_name:
            default_params = {"lr": 1e-3, "batch_size": 100, "z_dim": 20, "y_dim": 15, "s_dim": 20, "n_layers_surv_piecewise": 1, "n_intervals": 10}
        else:
            default_params = {"lr": 1e-3, "batch_size": 100, "z_dim": 20, "y_dim": 15, "s_dim": 20}
        if "_DP" in generator_name:
            default_params["max_grad_norm"] = 1.0
            default_params["epochs"] = 1000
        if tune_params is not None:
            default_params = {k: v for k, v in default_params.items() if k in tune_params}
        study.enqueue_trial(default_params)
        print("Enqueued trial:", study.get_trials(deepcopy=False))
    study.optimize(objective, n_trials=n_trials)
    study.best_params  

    return study.best_params, study

