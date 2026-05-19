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


warnings.filterwarnings("ignore")

def set_seed(seed=1):
    random.seed(seed)                            # Python built-in
    np.random.seed(seed)                         # NumPy
    torch.manual_seed(seed)                      # PyTorch (CPU)

# Standard HIVAE training loop.
# Splits `data` into 90/10 train/validation, freezes per-feature
# normalization statistics on the full training set and stores them on
# `vae_model.feat_normalization_globals` so every batch uses the same
# scale, then runs the optimizer for `epochs` with early stopping on the
# validation ELBO. The `seed` argument controls the split RNG and the
# rest of the loop's stochasticity.
def train_HIVAE(vae_model, data, miss_mask, true_miss_mask, feat_types_dict, batch_size, lr, epochs, verbose = True, seed=1, start_epoch=0):

    # Train-test split on control
    train_test_share = .9
    n_samples = data.shape[0]
    n_train_samples = int(train_test_share * n_samples)
    train_index = np.random.choice(n_samples, n_train_samples, replace=False)
    test_index = [i for i in np.arange(n_samples) if i not in train_index]

    data_train = data[train_index]
    miss_mask_train = miss_mask[train_index]
    true_miss_mask_train = true_miss_mask[train_index]

    data_test = data[test_index]
    miss_mask_test = miss_mask[test_index]
    true_miss_mask_test = true_miss_mask[test_index]

    # Number of batches
    n_train_samples = data_train.shape[0]
    if n_train_samples < batch_size:
        raise ValueError("Batch size must be less than the number of training samples")
    n_batches_train = int(np.floor(n_train_samples / batch_size))
    n_train_samples = n_batches_train * batch_size

    # Compute real missing mask
    miss_mask_train = torch.multiply(miss_mask_train, true_miss_mask_train)

    # On test/val
    n_test_samples = data_test.shape[0]
    # Adjust batch size if larger than dataset
    batch_test_size = n_test_samples
    # Number of batches
    n_batches_test = int(np.floor(n_test_samples / batch_test_size))

    # Compute real missing mask
    miss_mask_test = torch.multiply(miss_mask_test, true_miss_mask_test)
    # n_generated_sample = 10

    # ─── Freeze per-feature normalization globals on the training set ───
    vae_model.feat_normalization_globals = data_processing.compute_feat_normalization_globals(
        data_train, feat_types_dict, miss_mask_train
    )


    # Training
    optimizer = optim.Adam(vae_model.parameters(), lr=lr)

    start_time = time.time()
    loss_train, error_observed_train, error_missing_train = [], [], []
    loss_val, error_observed_val, error_missing_val = [], [], []

    rng = np.random.default_rng(seed=seed)
    # Setting for early stopping
    best_val_loss = float('inf')

    patience = 6
    n_iter_validation = 50
    n_iter_min = 100
    counter = 0
    # min_improvement_ratio = 0 #5e-3
    for epoch in range(epochs):
        global_epoch = epoch + start_epoch
        avg_loss, avg_KL_s, avg_KL_z = 0.0, 0.0, 0.0
        avg_loss_val, avg_KL_s_val, avg_KL_z_val = 0.0, 0.0, 0.0
        samples_list, p_params_list, q_params_list, log_p_x_total, log_p_x_missing_total = [], [], [], [], []
        tau = max(1.0 - 0.01 * global_epoch, 1e-3)

        # Shuffle training data
        perm = rng.permutation(data_train.shape[0])
        data_train = data_train[perm]
        miss_mask_train = miss_mask_train[perm]
        true_miss_mask_train = true_miss_mask_train[perm]

        for i in range(n_batches_train):
            # Get batch data
            data_list, miss_list = data_processing.next_batch(data_train, feat_types_dict, miss_mask_train, batch_size, i)

            # Mask unknown data (set unobserved values to zero)
            data_list_observed = [data * miss_list[:, i].view(batch_size, 1) for i, data in enumerate(data_list)]

            # Compute loss
            optimizer.zero_grad()
            vae_res = vae_model.forward(data_list_observed, data_list, miss_list, tau, n_generated_dataset=1)
            vae_res["neg_ELBO_loss"].backward()
            optimizer.step()

            avg_loss += vae_res["neg_ELBO_loss"].item() / n_batches_train
            avg_KL_s += torch.mean(vae_res["KL_s"]).item() / n_batches_train
            avg_KL_z += torch.mean(vae_res["KL_z"]).item() / n_batches_train

            # Save the generated samlpes and estimated parameters !
            samples_list.append(vae_res["samples"])
            p_params_list.append(vae_res["p_params"])
            q_params_list.append(vae_res["q_params"])
            log_p_x_total.append(vae_res["log_p_x"])
            log_p_x_missing_total.append(vae_res["log_p_x_missing"])

        # Concatenate samples in arrays
        s_total, z_total, y_total, est_data_train = statistic.samples_concatenation(samples_list)
        
        # Transform discrete variables back to the original values
        data_train_transformed = data_processing.discrete_variables_transformation(data_train[: n_train_samples], feat_types_dict)
        est_data_train_transformed = data_processing.discrete_variables_transformation(est_data_train[0], feat_types_dict)

        # Compute errors
        error_observed_samples, error_missing_samples = statistic.error_computation(data_train_transformed, est_data_train_transformed,
                                                                                    feat_types_dict, miss_mask_train[:n_train_samples])

        # Create global dictionary of the distribution parameters
        q_params_complete = statistic.q_distribution_params_concatenation(q_params_list)

        # Number of clusters created
        cluster_index = torch.argmax(q_params_complete['s'], 1)
        cluster = torch.unique(cluster_index)
        # print('Clusters: ' + str(len(cluster)))

        # Save average loss and error
        loss_train.append(avg_loss)
        error_observed_train.append(torch.mean(error_observed_samples))
        error_missing_train.append(torch.mean(error_missing_samples))
        if verbose:
            if global_epoch % 100 == 0:
                visualization.print_loss(global_epoch, start_time, -avg_loss, avg_KL_s, avg_KL_z)


        if epoch % n_iter_validation == 0:
            with torch.no_grad():
                for i in range(n_batches_test):
                    data_list_test, miss_list_test = data_processing.next_batch(data_test, feat_types_dict, miss_mask_test, batch_test_size, i)

                    # Mask unknown data (set unobserved values to zero)
                    data_list_observed_test = [data * miss_list_test[:, i].view(batch_test_size, 1) for i, data in enumerate(data_list_test)]

                    vae_res_test = vae_model.forward(data_list_observed_test, data_list_test, miss_list_test, tau=1e-3, n_generated_dataset=1)
                    avg_loss_val += vae_res_test["neg_ELBO_loss"].item() / n_batches_test
                    avg_KL_s_val += torch.mean(vae_res_test["KL_s"]).item() / n_batches_test
                    avg_KL_z_val += torch.mean(vae_res_test["KL_z"]).item() / n_batches_test

            loss_val.append(avg_loss_val)

            if avg_loss_val >= best_val_loss:
                counter += 1
            else:
                best_val_loss = avg_loss_val
                counter = 0

            if counter >= patience and epoch >= n_iter_min:
                print(f"Early stopping at epoch {global_epoch}.")
                break
        else:
            loss_val.append(torch.nan)

    if verbose:
        print("Training finished.")

    return vae_model, loss_train, loss_val


def generate_from_condition_HIVAE(vae_model, df, miss_mask, true_miss_mask, feat_types_dict, n_generated_dataset, n_generated_sample=None, from_prior=False, condition={'var': "treatment", 'value': 0.0, 'n_samples': 300}):
    
    data = torch.from_numpy(df.values)
    features = df.columns.tolist()
    cond_feature_idx = features.index(condition['var'])

    # Compute real missing mask
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

    

def generate_from_HIVAE(vae_model, data, miss_mask, true_miss_mask, feat_types_dict, n_generated_dataset, n_generated_sample=None, from_prior=False):

    # Compute real missing mask
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
        
        #Concatenate samples in arrays
        est_data_gen = statistic.samples_concatenation(samples_list)[-1]
        est_data_gen_transformed = []
        for j in range(n_generated_dataset):
            data_trans = data_processing.discrete_variables_transformation(est_data_gen[j], feat_types_dict)
            data_trans = data_processing.survival_variables_transformation(data_trans, feat_types_dict)
            est_data_gen_transformed.append(data_trans.unsqueeze(0))
            
        est_data_gen_transformed = torch.cat(est_data_gen_transformed, dim=0)

        return est_data_gen_transformed



# End-to-end entry point: build a HIVAE, train it on `df`, and generate
# `n_generated_dataset` synthetic datasets.
# The training routine is selected by the flags:
#   differential_privacy=True   -> train_HIVAE_DP
#   batchcorrect=True           -> train_HIVAE_bis
#   otherwise                   -> train_HIVAE
# `seed` is forwarded to set_seed and to the chosen train_* routine, so a
# single argument fully determines the run. `condition` (optional) routes
# generation through generate_from_condition_HIVAE; `gen_from_prior=True`
# samples z directly from the prior. With `plot=True`, the training/val
# loss curves are displayed at the end.
def run(df, miss_mask, true_miss_mask, feat_types_dict,  n_generated_dataset, n_generated_sample=None,
        params={"lr": 1e-3, "batch_size": 100, "z_dim": 20, "y_dim": 15, "s_dim": 20, "n_layers_surv_piecewise": 1, "n_intervals": 10},
        epochs=1000, verbose=True, plot=False, gen_from_prior=False, condition=None, differential_privacy=False, batchcorrect=False,
        seed=1,
        target_epsilon=None, target_delta=1e-5, max_grad_norm=1.0):

    set_seed(seed=seed)
    model_name = "HIVAE_inputDropout" # "HIVAE_factorized"

    miss_mask = miss_mask
    true_miss_mask = true_miss_mask
    dim_latent_z = params["z_dim"]
    dim_latent_y = params["y_dim"]
    dim_latent_s = params["s_dim"]
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
    if differential_privacy:
        if target_epsilon is None:
            raise ValueError("target_epsilon must be set when differential_privacy=True.")
        model_hivae, loss_train, loss_val, _ = train_HIVAE_DP(
            model_hivae, data, miss_mask, true_miss_mask, feat_types_dict,
            batch_size, lr, epochs,
            target_epsilon=target_epsilon, target_delta=target_delta, max_grad_norm=max_grad_norm,
            verbose=verbose, seed=seed,
        )
    else:
        if batchcorrect:
            model_hivae, loss_train, loss_val = train_HIVAE_bis(model_hivae, data, miss_mask, true_miss_mask, feat_types_dict, batch_size, lr, epochs, verbose, seed=seed)
        else:
            model_hivae, loss_train, loss_val = train_HIVAE(model_hivae, data, miss_mask, true_miss_mask, feat_types_dict, batch_size, lr, epochs, verbose, seed=seed)
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

        return est_data_gen_transformed
    
# Multi-run variant of run(): trains `n_runs` independent HIVAE models
# with seeds (seed, seed+1, …, seed+n_runs-1) and concatenates their
# generated datasets. Each model produces ~n_generated_dataset/n_runs
# datasets so the total output size matches a single run() call. Same
# flags as run() (differential_privacy / batchcorrect / condition /
# gen_from_prior) and uses the same train_HIVAE* dispatch. Returns a
# single tensor with all generated samples concatenated along dim 0.
def run_alt(df, miss_mask, true_miss_mask, feat_types_dict,  n_generated_dataset, n_generated_sample=None,
        params={"lr": 1e-3, "batch_size": 100, "z_dim": 20, "y_dim": 15, "s_dim": 20, "n_layers_surv_piecewise": 1, "n_intervals": 10},
        epochs=1000, verbose=True, plot=False, gen_from_prior=False, condition=None, differential_privacy=False, batchcorrect=False,
        seed=1, n_runs=5,
        target_epsilon=None, target_delta=1e-5, max_grad_norm=1.0):

    set_seed(seed=seed)
    model_name = "HIVAE_inputDropout" # "HIVAE_factorized"

    miss_mask = miss_mask
    true_miss_mask = true_miss_mask
    dim_latent_z = params["z_dim"]
    dim_latent_y = params["y_dim"]
    dim_latent_s = params["s_dim"]
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
    data = torch.from_numpy(df.values)

    n_generated_dataset_i = round(n_generated_dataset/n_runs)

    generated_dataset_list = []

    for run_i in range(n_runs):
        model_hivae = model_loading(input_dim=df.shape[1],
                                    z_dim=dim_latent_z,
                                    y_dim=dim_latent_y,
                                    s_dim=dim_latent_s, 
                                    y_dim_partition=None,
                                    feat_types_dict=feat_types_dict,
                                    intervals_surv_piecewise=intervals,
                                    n_layers_surv_piecewise=n_layers
                                    )
        if differential_privacy:
            if target_epsilon is None:
                raise ValueError("target_epsilon must be set when differential_privacy=True.")
            model_hivae, loss_train, loss_val, _ = train_HIVAE_DP(
                model_hivae, data, miss_mask, true_miss_mask, feat_types_dict,
                batch_size, lr, epochs,
                target_epsilon=target_epsilon, target_delta=target_delta, max_grad_norm=max_grad_norm,
                verbose=verbose, seed=seed+run_i,
            )
        else:
            if batchcorrect:
                model_hivae, loss_train, loss_val = train_HIVAE_bis(model_hivae, data, miss_mask, true_miss_mask, feat_types_dict, batch_size, lr, epochs, verbose, seed=seed+run_i)
            else:
                model_hivae, loss_train, loss_val = train_HIVAE(model_hivae, data, miss_mask, true_miss_mask, feat_types_dict, batch_size, lr, epochs, verbose, seed=seed+run_i)
        if isinstance(n_generated_sample, list):
            est_data_gen_transformed_list = []
            for n_generated_sample_ in n_generated_sample:
                if condition is not None:
                    est_data_gen_transformed = generate_from_condition_HIVAE(model_hivae, df, miss_mask, true_miss_mask, feat_types_dict, n_generated_dataset_i, n_generated_sample_, from_prior=gen_from_prior, condition=condition)
                else:
                    est_data_gen_transformed = generate_from_HIVAE(model_hivae, data, miss_mask, true_miss_mask, feat_types_dict, n_generated_dataset_i, n_generated_sample_, from_prior=gen_from_prior)
                est_data_gen_transformed_list.append(est_data_gen_transformed)

            return est_data_gen_transformed_list
        else:
            if condition is not None:
                est_data_gen_transformed = generate_from_condition_HIVAE(model_hivae, df, miss_mask, true_miss_mask, feat_types_dict, n_generated_dataset_i, n_generated_sample, from_prior=gen_from_prior, condition=condition)
            else:
                est_data_gen_transformed = generate_from_HIVAE(model_hivae, data, miss_mask, true_miss_mask, feat_types_dict, n_generated_dataset_i, n_generated_sample, from_prior=gen_from_prior)

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

            generated_dataset_list.append(est_data_gen_transformed)
    
    generated_datasets_torch = torch.cat(generated_dataset_list, dim=0)

    return generated_datasets_torch




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


def hyperparameter_space(data, n_splits, generator_name):
    """
    Define the hyperparameter space for the model

    Parameters to optimize: z_dim, y_dim, s_dim, batch_size, lr, n_layers_surv_piecewise
    """
    n_samples = data.shape[0]
    hp_space = [
        CategoricalDistribution(name="lr", choices=[1e-4, 2e-4, 1e-3, 2e-3, 3e-3, 5e-3]),
        CategoricalDistribution(name="batch_size", choices=get_batchsize(n_samples, n_splits) + [32, 100]),
        IntegerDistribution(name="z_dim", low=10, high=200, step=10),
        IntegerDistribution(name="y_dim", low=10, high=200, step=10),
        IntegerDistribution(name="s_dim", low=10, high=200, step=10),
    ]
    if "HI-VAE_piecewise" in generator_name:
       hp_space.append(CategoricalDistribution(name="n_layers_surv_piecewise", choices=[1, 2]))
       hp_space.append(CategoricalDistribution(name="n_intervals", choices=[5, 10, 15, 20]))

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

def optuna_hyperparameter_search(df, miss_mask, true_miss_mask, feat_types_dict, n_generated_dataset, n_splits, n_trials, columns, generator_name, epochs = 1000, n_generated_sample = None, study_name='optuna_study_surv_hivae', metric='survival_km_distance', method='', gen_from_prior=False, condition=None, cond_df=None, batchcorrect=False, seed=10,
                                 target_epsilon=None, target_delta=1e-5, max_grad_norm=1.0):
    if "_DP" in generator_name and target_epsilon is None:
        raise ValueError("target_epsilon must be set when generator_name contains '_DP'.")
   
    model_name = "HIVAE_inputDropout" # "HIVAE_factorized"
    miss_mask = miss_mask
    true_miss_mask = true_miss_mask
    if condition is not None and cond_df is not None:
        cond_full_data_loader =  SurvivalAnalysisDataLoader(cond_df, target_column = "censor", time_to_event_column = "time")
 
    def objective(trial: optuna.Trial):
        set_seed()
        hp_space = hyperparameter_space(df, n_splits, generator_name)
        params = suggest_all(trial, hp_space) # dict of hyperparameters
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

                if "_DP" in generator_name:
                    model_hivae, _, _, _ = train_HIVAE_DP(
                        model_hivae, data, miss_mask, true_miss_mask, feat_types_dict,
                        batch_size, params["lr"], epochs,
                        target_epsilon=target_epsilon, target_delta=target_delta, max_grad_norm=max_grad_norm,
                    )
                else:
                    if batchcorrect:
                        model_hivae, _, _ = train_HIVAE_bis(model_hivae, data, miss_mask, true_miss_mask, feat_types_dict, batch_size, params["lr"], epochs)
                    else:
                        model_hivae, _, _ = train_HIVAE(model_hivae, data, miss_mask, true_miss_mask, feat_types_dict, batch_size, params["lr"], epochs)
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
                model_hivae, _, _ = train_HIVAE(model_hivae, train_data, train_miss_mask, train_true_miss_mask, feat_types_dict, batch_size, params["lr"], epochs)
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
                model_hivae, _, _ = train_HIVAE(model_hivae, train_data, train_miss_mask, train_true_miss_mask, feat_types_dict, batch_size, params["lr"], epochs)
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
            
                # # k-fold cross-validation
                # kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                # for train_index, test_index in kf.split(data):
                #     train_data, test_data = data[train_index], data[test_index]
                #     df_test_data = df.iloc[test_index]
                #     test_data_loader = SurvivalAnalysisDataLoader(df_test_data, target_column = "censor", time_to_event_column = "time")
                #     full_data_loader = SurvivalAnalysisDataLoader(df, target_column = "censor", time_to_event_column = "time")
                #     train_miss_mask, test_miss_mask = miss_mask[train_index], miss_mask[test_index]
                #     train_true_miss_mask, test_true_miss_mask = true_miss_mask[train_index], true_miss_mask[test_index]
                    
                #     if method == 'train_train_gen_full':
                #         # Train
                #         batch_size = params["batch_size"]
                #         batch_size = min(batch_size, data.shape[0])
                #         model_hivae = model_loading(input_dim=data.shape[1],
                #                     z_dim=params["z_dim"],
                #                     y_dim=params["y_dim"],
                #                     s_dim=params["s_dim"],
                #                     y_dim_partition=None,
                #                     feat_types_dict=feat_types_dict,
                #                     intervals_surv_piecewise=intervals,
                #                     n_layers_surv_piecewise=n_layers)
                #         model_hivae, _, _ = train_HIVAE(model_hivae, train_data, train_miss_mask, train_true_miss_mask, feat_types_dict, batch_size, params["lr"], epochs)
                #         # Generate
                #         if condition is not None:
                #             est_data_gen_transformed = generate_from_condition_HIVAE(model_hivae, df, miss_mask, true_miss_mask,
                #                                                                     feat_types_dict, n_generated_dataset, n_generated_sample=data.shape[0], from_prior=gen_from_prior, condition=condition)
                #         else:
                #             est_data_gen_transformed = generate_from_HIVAE(model_hivae, data, miss_mask, true_miss_mask,
                #                                                         feat_types_dict, n_generated_dataset=n_generated_dataset, n_generated_sample=data.shape[0], from_prior=gen_from_prior)
                #         score_k = []
                #         for j in range(n_generated_dataset):
                #             df_gen_data = pd.DataFrame(est_data_gen_transformed[j].numpy(), columns=columns)
                #             if metric == 'log_rank_test':
                #                 score_kj = metrics.compute_logrank_test(df, df_gen_data)
                #             else: # 'survival_km_distance'
                #                 gen_data = SurvivalAnalysisDataLoader(df_gen_data, target_column = "censor", time_to_event_column = "time")
                #                 clear_cache()
                #                 evaluation = Metrics().evaluate(X_gt=full_data_loader, # can be dataloaders or dataframes
                #                                                 X_syn=gen_data, 
                #                                                 reduction='mean', # default mean
                #                                                 n_histogram_bins=10, # default 10
                #                                                 n_folds=1,
                #                                                 metrics={'stats': ['survival_km_distance']},
                #                                                 task_type='survival_analysis', 
                #                                                 use_cache=True)
                #                 score_kj = evaluation.T[["stats.survival_km_distance.abs_optimism"]].T["mean"].values[0]
                #             score_k.append(score_kj)

                #     else:
                #         # Train
                #         batch_size = params["batch_size"]
                #         batch_size = min(batch_size, data.shape[0])
                #         model_hivae = model_loading(input_dim=data.shape[1],
                #                     z_dim=params["z_dim"],
                #                     y_dim=params["y_dim"],
                #                     s_dim=params["s_dim"],
                #                     y_dim_partition=None,
                #                     feat_types_dict=feat_types_dict,
                #                     intervals_surv_piecewise=intervals,
                #                     n_layers_surv_piecewise=n_layers)
                #         model_hivae, _, _ = train_HIVAE(model_hivae, train_data, train_miss_mask, train_true_miss_mask, feat_types_dict, batch_size, params["lr"], epochs)
                #         # Generate
                #         est_data_gen_transformed = generate_from_HIVAE(model_hivae, test_data, test_miss_mask, test_true_miss_mask,
                #                                                         feat_types_dict, n_generated_dataset=n_generated_dataset, n_generated_sample=test_data.shape[0], from_prior=gen_from_prior)

                #         score_k = []
                #         for j in range(n_generated_dataset):
                #             df_gen_data = pd.DataFrame(est_data_gen_transformed[j].numpy(), columns=columns)
                #             if metric == 'log_rank_test':
                #                 score_kj = metrics.compute_logrank_test(df_test_data, df_gen_data)
                #             else: # 'survival_km_distance'
                #                 gen_data = SurvivalAnalysisDataLoader(df_gen_data, target_column = "censor", time_to_event_column = "time")
                #                 clear_cache()
                #                 evaluation = Metrics().evaluate(X_gt=test_data_loader, # can be dataloaders or dataframes
                #                                                 X_syn=gen_data, 
                #                                                 reduction='mean', # default mean
                #                                                 n_histogram_bins=10, # default 10
                #                                                 n_folds=1,
                #                                                 metrics={'stats': ['survival_km_distance']},
                #                                                 task_type='survival_analysis', 
                #                                                 use_cache=True)
                #                 score_kj = evaluation.T[["stats.survival_km_distance.abs_optimism"]].T["mean"].values[0]
                #             score_k.append(score_kj)
                #     scores.append(np.mean(score_k))
            print(f"Score: {np.mean(scores)}")
        except Exception as e:  # invalid set of params
            print(f"{type(e).__name__}: {e}")
            print(params)
            raise optuna.TrialPruned()
        return np.mean(scores)
    

    db_file = study_name + '.db'
    if os.path.exists(db_file):
        print("This optuna study ({}) already exists. We load the study from the existing file.".format(db_file))
        study = optuna.load_study(study_name=study_name, storage='sqlite:///'+study_name+'.db')
    else: 
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="minimize", study_name=study_name, storage='sqlite:///'+study_name+'.db', sampler=sampler)
        if "HI-VAE_piecewise" in generator_name:
            default_params = {"lr": 1e-3, "batch_size": 100, "z_dim": 20, "y_dim": 15, "s_dim": 20, "n_layers_surv_piecewise": 1, "n_intervals": 10}
        else: 
            default_params = {"lr": 1e-3, "batch_size": 100, "z_dim": 20, "y_dim": 15, "s_dim": 20}
        study.enqueue_trial(default_params)
        print("Enqueued trial:", study.get_trials(deepcopy=False))
    study.optimize(objective, n_trials=n_trials)
    study.best_params  

    return study.best_params, study





def run_CV(df, miss_mask, true_miss_mask, feat_types_dict, n_generated_dataset, n_splits, n_trials, columns, generator_name, epochs = 1000, study_name='optuna_study_surv_hivae', metric='survival_km_distance', params={}):
   
    set_seed()
    model_name = "HIVAE_inputDropout" # "HIVAE_factorized"
    miss_mask = miss_mask
    true_miss_mask = true_miss_mask
        
    if "HI-VAE_piecewise" in generator_name:
        intervals = get_intervals(df, params["n_intervals"])
        n_layers = params["n_layers_surv_piecewise"]
    else:
        intervals = None
        n_layers = None
    model_loading = getattr(importlib.import_module("src"), model_name)
    data = torch.from_numpy(df.values)
    scores = []
    # k-fold cross-validation
    # kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    kf = KFold(n_splits=n_splits, shuffle=False)
    for train_index, test_index in kf.split(data):
        train_data, test_data = data[train_index], data[test_index]
        df_test_data = df.iloc[test_index]
        test_data_loader = SurvivalAnalysisDataLoader(df_test_data, target_column = "censor", time_to_event_column = "time")
        train_miss_mask, test_miss_mask = miss_mask[train_index], miss_mask[test_index]
        train_true_miss_mask, test_true_miss_mask = true_miss_mask[train_index], true_miss_mask[test_index]
        
        # Train
        batch_size = params["batch_size"]
        batch_size = min(batch_size, data.shape[0])
        model_hivae = model_loading(input_dim=data.shape[1],
                    z_dim=params["z_dim"],
                    y_dim=params["y_dim"],
                    s_dim=params["s_dim"],
                    y_dim_partition=None,
                    feat_types_dict=feat_types_dict,
                    intervals_surv_piecewise=intervals,
                    n_layers_surv_piecewise=n_layers)
        model_hivae, _, _ = train_HIVAE(model_hivae, train_data, train_miss_mask, train_true_miss_mask, feat_types_dict, batch_size, params["lr"], epochs)
        # Generate
        est_data_gen_transformed = generate_from_HIVAE(model_hivae, test_data, test_miss_mask, test_true_miss_mask,
                                                        feat_types_dict, n_generated_dataset=n_generated_dataset, n_generated_sample=test_data.shape[0])

        score_k = []
        for j in range(n_generated_dataset):
            df_gen_data = pd.DataFrame(est_data_gen_transformed[j].numpy(), columns=columns)
            df_gen_data["time"] = df_gen_data["time"].clip(lower=1e-6)
            if metric == 'log_rank_test':
                score_kj = metrics.compute_logrank_test(df_test_data, df_gen_data)
            else: # 'survival_km_distance'
                gen_data = SurvivalAnalysisDataLoader(df_gen_data, target_column = "censor", time_to_event_column = "time")
                clear_cache()
                evaluation = Metrics().evaluate(X_gt=test_data_loader, # can be dataloaders or dataframes
                                                X_syn=gen_data, 
                                                reduction='mean', # default mean
                                                n_histogram_bins=10, # default 10
                                                n_folds=1,
                                                metrics={'stats': ['survival_km_distance']},
                                                task_type='survival_analysis', 
                                                use_cache=True)
                score_kj = evaluation.T[["stats.survival_km_distance.abs_optimism"]].T["mean"].values[0]
            score_k.append(score_kj)
        scores.append(np.mean(score_k))
    print(f"Score: {np.mean(scores)}")
    return np.mean(scores)



from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

# HIVAE training loop with differential privacy via opacus.PrivacyEngine.
# Same 90/10 split and frozen per-feature normalization globals as
# train_HIVAE, but the optimizer is wrapped with make_private_with_epsilon
# so per-sample gradients are clipped to `max_grad_norm` and Gaussian
# noise is added before each parameter update, calibrated to land at
# ε ≤ target_epsilon at the given δ after `epochs` epochs at sample
# rate batch_size / n_train.
#
# Differences vs train_HIVAE:
#   * `target_epsilon` (required), `target_delta`, `max_grad_norm` are
#     new arguments. The privacy budget is honoured at the end.
#   * Early stopping is disabled — σ is calibrated for the full schedule,
#     so stopping early would over-noise without spending the budget.
#     Use train_HIVAE's early-stopping epoch as a proxy when choosing
#     `epochs`.
#   * Validation loss is still computed every n_iter_validation epochs
#     for monitoring/plotting, but never influences training length.
#   * Returns (vae_model, loss_train, loss_val, final_epsilon). The
#     returned vae_model is the *unwrapped* HIVAE — Opacus's
#     GradSampleModule has been stripped and the model is in eval()
#     mode, so the generate_* functions can use it without knowing
#     anything about Opacus. `final_epsilon` is the budget consumed
#     at target_delta as reported by the privacy accountant.
#   * `seed` seeds the train/test split, the DataLoader shuffling and
#     Opacus's noise generator (for reproducibility — strict DP wants
#     fresh randomness; this is the standard research compromise).
def train_HIVAE_DP(vae_model, data, miss_mask, true_miss_mask, feat_types_dict,
                   batch_size, lr, epochs,
                   target_epsilon, target_delta=1e-5, max_grad_norm=1.0,
                   verbose=True, seed=1, start_epoch=0):

    # ─── Opacus compatibility check ────────────────────────────────────
    # Catches unsupported layers (BatchNorm, LSTM, …) before training. It
    # does NOT catch in-forward sample coupling (e.g. computing batch
    # statistics inside a preprocessing step), which is why the model
    # also relies on feat_normalization_globals being frozen below.
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

    # ─── Train/test split (seeded) ─────────────────────────────────────
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

    n_train_samples = data_train.shape[0]
    if n_train_samples < batch_size:
        raise ValueError("Batch size must be less than the number of training samples")

    # Combined real-missing masks
    miss_mask_train = torch.multiply(miss_mask_train, true_miss_mask_train)
    miss_mask_test = torch.multiply(miss_mask_test, true_miss_mask_test)

    # Test/val batching (single full-batch pass)
    n_test_samples = data_test.shape[0]
    batch_test_size = n_test_samples
    n_batches_test = int(np.floor(n_test_samples / batch_test_size))

    # ─── Freeze per-feature normalization globals on the training set ───
    vae_model.feat_normalization_globals = data_processing.compute_feat_normalization_globals(
        data_train, feat_types_dict, miss_mask_train
    )

    # ─── Optimizer, dataset, loader ────────────────────────────────────
    optimizer = optim.Adam(vae_model.parameters(), lr=lr)
    dataset = MyCustomDataset(data_train, miss_mask_train, feat_types_dict)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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

    # ─── Training loop (no early stopping) ─────────────────────────────
    start_time = time.time()
    loss_train, loss_val = [], []
    n_iter_validation = 50

    for epoch in range(epochs):
        global_epoch = epoch + start_epoch
        avg_loss = avg_KL_s = avg_KL_z = 0.0
        avg_loss_val = avg_KL_s_val = avg_KL_z_val = 0.0
        tau = max(1.0 - 0.01 * global_epoch, 1e-3)

        n_batches_seen = 0
        for batch_data_list, batch_miss_list in train_loader:
            # Mask unknown data (set unobserved values to zero). Under
            # Poisson sampling each batch's row count varies, so use the
            # batch tensor's own shape rather than a precomputed value.
            data_list_observed = [
                feat * batch_miss_list[:, j].view(feat.shape[0], 1)
                for j, feat in enumerate(batch_data_list)
            ]

            optimizer.zero_grad()
            vae_res = vae_model.forward(
                data_list_observed, batch_data_list, batch_miss_list,
                tau, n_generated_dataset=1,
            )
            vae_res["neg_ELBO_loss"].backward()
            optimizer.step()

            avg_loss += vae_res["neg_ELBO_loss"].item()
            avg_KL_s += torch.mean(vae_res["KL_s"]).item()
            avg_KL_z += torch.mean(vae_res["KL_z"]).item()
            n_batches_seen += 1

        # Average over batches actually seen — DPDataLoader uses Poisson
        # sampling so the per-epoch batch count is random; dividing by a
        # precomputed n_train/batch_size would bias the reported average.
        if n_batches_seen > 0:
            avg_loss /= n_batches_seen
            avg_KL_s /= n_batches_seen
            avg_KL_z /= n_batches_seen

        # Note: per-epoch reconstruction error / cluster diagnostics are
        # not computed in DP mode. Under Poisson subsampling the
        # concatenated batch samples are not aligned with
        # data_train[:n_train_samples] (the non-DP loop maintains this
        # alignment via explicit permutation; DPDataLoader does not),
        # so error_computation against indexed training data would
        # produce meaningless numbers.

        loss_train.append(avg_loss)

        if verbose and global_epoch % 100 == 0:
            visualization.print_loss(global_epoch, start_time, -avg_loss, avg_KL_s, avg_KL_z)

        # Validation pass — for monitoring only, not used for early stopping.
        if epoch % n_iter_validation == 0:
            with torch.no_grad():
                for i in range(n_batches_test):
                    data_list_test, miss_list_test = data_processing.next_batch(
                        data_test, feat_types_dict, miss_mask_test, batch_test_size, i,
                    )
                    data_list_observed_test = [
                        feat * miss_list_test[:, j].view(batch_test_size, 1)
                        for j, feat in enumerate(data_list_test)
                    ]
                    vae_res_test = vae_model.forward(
                        data_list_observed_test, data_list_test, miss_list_test,
                        tau=1e-3, n_generated_dataset=1,
                    )
                    avg_loss_val += vae_res_test["neg_ELBO_loss"].item() / max(n_batches_test, 1)
                    avg_KL_s_val += torch.mean(vae_res_test["KL_s"]).item() / max(n_batches_test, 1)
                    avg_KL_z_val += torch.mean(vae_res_test["KL_z"]).item() / max(n_batches_test, 1)
            loss_val.append(avg_loss_val)
        else:
            loss_val.append(torch.nan)

    # ─── Report consumed privacy budget ────────────────────────────────
    final_epsilon = privacy_engine.get_epsilon(delta=target_delta)

    if verbose:
        print(f"Training finished. ε = {final_epsilon:.3f} at δ = {target_delta}.")

    # ─── Unwrap the model for downstream generation ────────────────────
    # GradSampleModule wraps the trained HIVAE; the inner module carries
    # the trained parameters and feat_normalization_globals attribute.
    # Switch to eval() so subsequent forward passes are deterministic.
    unwrapped_model = vae_model._module
    unwrapped_model.eval()

    return unwrapped_model, loss_train, loss_val, final_epsilon


# Variant of the HIVAE training loop used for the batch-corrected setup
# (selected by `batchcorrect=True` in run()). Same 90/10 split, frozen
# per-feature normalization globals on the training set, and early
# stopping as train_HIVAE; the body differs in how each batch's
# reconstruction loss is aggregated to mitigate batch effects across the
# heterogeneous feature types. The `seed` argument controls split and
# loop RNG.
def train_HIVAE_bis(vae_model, data, miss_mask, true_miss_mask, feat_types_dict, batch_size, lr, epochs, verbose = True, seed=1, start_epoch=0):

    # Train-test split on control
    train_test_share = .9
    n_samples = data.shape[0]
    n_train_samples = int(train_test_share * n_samples)
    train_index = np.random.choice(n_samples, n_train_samples, replace=False)
    test_index = [i for i in np.arange(n_samples) if i not in train_index]

    data_train = data[train_index]
    miss_mask_train = miss_mask[train_index]
    true_miss_mask_train = true_miss_mask[train_index]

    data_test = data[test_index]
    miss_mask_test = miss_mask[test_index]
    true_miss_mask_test = true_miss_mask[test_index]

    # Number of batches
    n_train_samples = data_train.shape[0]
    if n_train_samples < batch_size:
        raise ValueError("Batch size must be less than the number of training samples")
    n_batches_train = int(np.floor(n_train_samples / batch_size))
    # n_train_samples = n_batches_train * batch_size

    # Compute real missing mask
    miss_mask_train = torch.multiply(miss_mask_train, true_miss_mask_train)

    # On test/val
    n_test_samples = data_test.shape[0]
    # Adjust batch size if larger than dataset
    batch_test_size = n_test_samples
    # Number of batches
    n_batches_test = int(np.floor(n_test_samples / batch_test_size))

    # Compute real missing mask
    miss_mask_test = torch.multiply(miss_mask_test, true_miss_mask_test)
    # n_generated_sample = 10

    # ─── Freeze per-feature normalization globals on the training set ───
    vae_model.feat_normalization_globals = data_processing.compute_feat_normalization_globals(
        data_train, feat_types_dict, miss_mask_train
    )


    # Training
    optimizer = optim.Adam(vae_model.parameters(), lr=lr)

    start_time = time.time()
    loss_train, error_observed_train, error_missing_train = [], [], []
    loss_val, error_observed_val, error_missing_val = [], [], []

    rng = np.random.default_rng(seed=seed)
    # Setting for early stopping
    best_val_loss = float('inf')

    patience = 6
    n_iter_validation = 50
    n_iter_min = 100
    counter = 0

    dataset = MyCustomDataset(data_train, miss_mask_train, feat_types_dict)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) #, drop_last=True)

    for epoch in range(epochs):
        global_epoch = epoch + start_epoch

        avg_loss, avg_KL_s, avg_KL_z = 0.0, 0.0, 0.0
        avg_loss_val, avg_KL_s_val, avg_KL_z_val = 0.0, 0.0, 0.0
        samples_list, p_params_list, q_params_list, log_p_x_total, log_p_x_missing_total = [], [], [], [], []
        tau = max(1.0 - 0.01 * global_epoch, 1e-3)

        for batch_data_list, batch_miss_list in train_loader:

            # Mask unknown data (set unobserved values to zero)
            data_list_observed = [data * batch_miss_list[:, i].view(data.shape[0], 1) for i, data in enumerate(batch_data_list)]
            # data_list_observed = [data * miss[:, None] for data, miss in zip(batch_data_list, batch_miss_list)]

            # Compute loss
            optimizer.zero_grad()
            vae_res = vae_model.forward(data_list_observed, batch_data_list, batch_miss_list, tau, n_generated_dataset=1)
            vae_res["neg_ELBO_loss"].backward()
            optimizer.step()

            avg_loss += vae_res["neg_ELBO_loss"].item() / n_batches_train
            avg_KL_s += torch.mean(vae_res["KL_s"]).item() / n_batches_train
            avg_KL_z += torch.mean(vae_res["KL_z"]).item() / n_batches_train

            # Save the generated samlpes and estimated parameters !
            samples_list.append(vae_res["samples"])
            p_params_list.append(vae_res["p_params"])
            q_params_list.append(vae_res["q_params"])
            log_p_x_total.append(vae_res["log_p_x"])
            log_p_x_missing_total.append(vae_res["log_p_x_missing"])

        # Concatenate samples in arrays
        s_total, z_total, y_total, est_data_train = statistic.samples_concatenation(samples_list)

        n_train_samples = min(est_data_train[0].shape[0], data_train.shape[0])
        # Transform discrete variables back to the original values
        data_train_transformed = data_processing.discrete_variables_transformation(data_train[:n_train_samples], feat_types_dict)
        est_data_train_transformed = data_processing.discrete_variables_transformation(est_data_train[0][:n_train_samples], feat_types_dict)

        # Compute errors
        error_observed_samples, error_missing_samples = statistic.error_computation(data_train_transformed, est_data_train_transformed,
                                                                                    feat_types_dict,
                                                                                    miss_mask_train[:n_train_samples])
        
        # Create global dictionary of the distribution parameters
        q_params_complete = statistic.q_distribution_params_concatenation(q_params_list)
        
        # Number of clusters created
        cluster_index = torch.argmax(q_params_complete['s'], 1)
        cluster = torch.unique(cluster_index)
        # print('Clusters: ' + str(len(cluster)))

        # Save average loss and error
        loss_train.append(avg_loss)
        error_observed_train.append(torch.mean(error_observed_samples))
        error_missing_train.append(torch.mean(error_missing_samples))
        if verbose:
            if global_epoch % 100 == 0:
                visualization.print_loss(global_epoch, start_time, -avg_loss, avg_KL_s, avg_KL_z)


        if epoch % n_iter_validation == 0:
            with torch.no_grad():            
                for i in range(n_batches_test):
                    data_list_test, miss_list_test = data_processing.next_batch(data_test, feat_types_dict, miss_mask_test, batch_test_size, i)
                
                    # Mask unknown data (set unobserved values to zero)
                    data_list_observed_test = [data * miss_list_test[:, i].view(batch_test_size, 1) for i, data in enumerate(data_list_test)]
                
                    vae_res_test = vae_model.forward(data_list_observed_test, data_list_test, miss_list_test, tau=1e-3, n_generated_dataset=1)
                    avg_loss_val += vae_res_test["neg_ELBO_loss"].item() / n_batches_test
                    avg_KL_s_val += torch.mean(vae_res_test["KL_s"]).item() / n_batches_test
                    avg_KL_z_val += torch.mean(vae_res_test["KL_z"]).item() / n_batches_test
            
            loss_val.append(avg_loss_val)

            if avg_loss_val >= best_val_loss:
                counter += 1
            else: 
                best_val_loss = avg_loss_val
                counter = 0

            if counter >= patience and epoch >= n_iter_min:
                print(f"Early stopping at epoch {global_epoch}.")
                break
        else:
            loss_val.append(torch.nan)

    if verbose:
        print("Training finished.")

    return vae_model, loss_train, loss_val

# Optuna-driven hyperparameter search for HIVAE.
# Each trial samples a hyperparameter configuration from
# hyperparameter_space(...) and evaluates it according to one of three
# train/generate/score schemes selected by `method`:
#   'train_full_gen_full'  : train on all data; score generated vs. full data.
#   'train_train_gen_full' : 80/20 split; train on 80; score gen vs. full.
#   'train_train_gen_test' : 80/20 split; train on 80; score gen vs. the 20.
# In each scheme two HIVAE models are trained with seeds (seed, seed+1)
# and each generates half of n_generated_dataset; their outputs are
# concatenated and scored once via synthcity's survival_km_distance. Time
# values are clipped to >= 1e-6 before being wrapped in a
# SurvivalAnalysisDataLoader so KM-based metrics don't choke on
# non-positive samples. The Optuna study is persisted to a SQLite file
# at `study_path + study_name + '.db'` and resumed if it already exists;
# otherwise a TPE sampler seeded with `seed` is used and a default-params
# trial is enqueued first. Returns (best_params, study).
def optuna_hyperparameter_search_alt(df, miss_mask, true_miss_mask, feat_types_dict, n_generated_dataset, n_splits, n_trials, columns, generator_name,
                                     epochs = 1000, n_generated_sample = None,study_name='optuna_study_surv_hivae', study_path = '', metric='survival_km_distance',
                                     method='', gen_from_prior=False, condition=None, cond_df=None,batchcorrect=False, seed=10,
                                     target_epsilon=None, target_delta=1e-5, max_grad_norm=1.0):
    if "_DP" in generator_name and target_epsilon is None:
        raise ValueError("target_epsilon must be set when generator_name contains '_DP'.")
   
    model_name = "HIVAE_inputDropout" # "HIVAE_factorized"
    miss_mask = miss_mask
    true_miss_mask = true_miss_mask
    if condition is not None and cond_df is not None:
        cond_full_data_loader =  SurvivalAnalysisDataLoader(cond_df, target_column = "censor", time_to_event_column = "time")
 
    def objective(trial: optuna.Trial):
        set_seed()
        hp_space = hyperparameter_space(df, n_splits, generator_name)
        params = suggest_all(trial, hp_space) # dict of hyperparameters
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
        scores = []
        try:
            if method == 'train_full_gen_full':

                full_data_loader = SurvivalAnalysisDataLoader(df, target_column = "censor", time_to_event_column = "time")
                # Train
                batch_size = params["batch_size"]
                batch_size = min(batch_size, int(0.9*data.shape[0]))
                model_hivae_1 = model_loading(input_dim=data.shape[1],
                                              z_dim=params["z_dim"],
                                              y_dim=params["y_dim"],
                                              s_dim=params["s_dim"],
                                              y_dim_partition=None,
                                              feat_types_dict=feat_types_dict,
                                              intervals_surv_piecewise=intervals,
                                              n_layers_surv_piecewise=n_layers)
                model_hivae_2 = model_loading(input_dim=data.shape[1],
                                              z_dim=params["z_dim"],
                                              y_dim=params["y_dim"],
                                              s_dim=params["s_dim"],
                                              y_dim_partition=None,
                                              feat_types_dict=feat_types_dict,
                                              intervals_surv_piecewise=intervals,
                                              n_layers_surv_piecewise=n_layers)                

            

                if "_DP" in generator_name:
                    model_hivae_1, _, _, _ = train_HIVAE_DP(
                        model_hivae_1, data, miss_mask, true_miss_mask, feat_types_dict,
                        batch_size, params["lr"], epochs,
                        target_epsilon=target_epsilon, target_delta=target_delta, max_grad_norm=max_grad_norm,
                        seed=seed,
                    )
                    model_hivae_2, _, _, _ = train_HIVAE_DP(
                        model_hivae_2, data, miss_mask, true_miss_mask, feat_types_dict,
                        batch_size, params["lr"], epochs,
                        target_epsilon=target_epsilon, target_delta=target_delta, max_grad_norm=max_grad_norm,
                        seed=seed+1,
                    )
                else:
                    if batchcorrect:
                        model_hivae_1, _, _ = train_HIVAE_bis(model_hivae_1, data, miss_mask, true_miss_mask, feat_types_dict, batch_size, params["lr"], epochs, seed=seed)
                        model_hivae_2, _, _ = train_HIVAE_bis(model_hivae_2, data, miss_mask, true_miss_mask, feat_types_dict, batch_size, params["lr"], epochs, seed=seed+1)

                    else:
                        model_hivae_1, _, _ = train_HIVAE(model_hivae_1, data, miss_mask, true_miss_mask, feat_types_dict, batch_size, params["lr"], epochs, seed=seed)
                        model_hivae_2, _, _ = train_HIVAE(model_hivae_2, data, miss_mask, true_miss_mask, feat_types_dict, batch_size, params["lr"], epochs, seed=seed+1)
                # Generate
                if condition is not None:
                    est_data_gen_transformed_1 = generate_from_condition_HIVAE(model_hivae_1, df, miss_mask, true_miss_mask,
                                                                            feat_types_dict, n_generated_dataset//2, n_generated_sample=data.shape[0], from_prior=gen_from_prior, condition=condition)
                    est_data_gen_transformed_2 = generate_from_condition_HIVAE(model_hivae_2, df, miss_mask, true_miss_mask,
                                                                            feat_types_dict, n_generated_dataset - n_generated_dataset//2, n_generated_sample=data.shape[0], from_prior=gen_from_prior, condition=condition)

                    est_data_gen_transformed = torch.cat([est_data_gen_transformed_1, est_data_gen_transformed_2], dim=0)

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
                    est_data_gen_transformed_1 = generate_from_HIVAE(model_hivae_1, data, miss_mask, true_miss_mask,
                                                                     feat_types_dict, n_generated_dataset//2, n_generated_sample=n_gen_sample, from_prior=gen_from_prior)
                    est_data_gen_transformed_2 = generate_from_HIVAE(model_hivae_2, data, miss_mask, true_miss_mask,
                                                                     feat_types_dict, n_generated_dataset - n_generated_dataset//2, n_generated_sample=n_gen_sample, from_prior=gen_from_prior)

                    est_data_gen_transformed = torch.cat([est_data_gen_transformed_1, est_data_gen_transformed_2], dim=0)

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
                model_hivae_1 = model_loading(input_dim=data.shape[1],
                            z_dim=params["z_dim"],
                            y_dim=params["y_dim"],
                            s_dim=params["s_dim"],
                            y_dim_partition=None,
                            feat_types_dict=feat_types_dict,
                            intervals_surv_piecewise=intervals,
                            n_layers_surv_piecewise=n_layers)
                model_hivae_2 = model_loading(input_dim=data.shape[1],
                            z_dim=params["z_dim"],
                            y_dim=params["y_dim"],
                            s_dim=params["s_dim"],
                            y_dim_partition=None,
                            feat_types_dict=feat_types_dict,
                            intervals_surv_piecewise=intervals,
                            n_layers_surv_piecewise=n_layers)
                model_hivae_1, _, _ = train_HIVAE(model_hivae_1, train_data, train_miss_mask, train_true_miss_mask, feat_types_dict, batch_size, params["lr"], epochs, seed=seed)
                model_hivae_2, _, _ = train_HIVAE(model_hivae_2, train_data, train_miss_mask, train_true_miss_mask, feat_types_dict, batch_size, params["lr"], epochs, seed=seed+1)
                # Generate
                if condition is not None:
                    est_data_gen_transformed_1 = generate_from_condition_HIVAE(model_hivae_1, df, miss_mask, true_miss_mask,
                                                                            feat_types_dict, n_generated_dataset//2, n_generated_sample=data.shape[0], from_prior=gen_from_prior, condition=condition)
                    est_data_gen_transformed_2 = generate_from_condition_HIVAE(model_hivae_2, df, miss_mask, true_miss_mask,
                                                                            feat_types_dict, n_generated_dataset - n_generated_dataset//2, n_generated_sample=data.shape[0], from_prior=gen_from_prior, condition=condition)
                    est_data_gen_transformed = torch.cat([est_data_gen_transformed_1, est_data_gen_transformed_2], dim=0)
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
                    est_data_gen_transformed_1 = generate_from_HIVAE(model_hivae_1, data, miss_mask, true_miss_mask,
                                                                feat_types_dict, n_generated_dataset=n_generated_dataset//2, n_generated_sample=data.shape[0], from_prior=gen_from_prior)
                    est_data_gen_transformed_2 = generate_from_HIVAE(model_hivae_2, data, miss_mask, true_miss_mask,
                                                                feat_types_dict, n_generated_dataset=n_generated_dataset - n_generated_dataset//2, n_generated_sample=data.shape[0], from_prior=gen_from_prior)
                    est_data_gen_transformed = torch.cat([est_data_gen_transformed_1, est_data_gen_transformed_2], dim=0)
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
                model_hivae_1 = model_loading(input_dim=data.shape[1],
                            z_dim=params["z_dim"],
                            y_dim=params["y_dim"],
                            s_dim=params["s_dim"],
                            y_dim_partition=None,
                            feat_types_dict=feat_types_dict,
                            intervals_surv_piecewise=intervals,
                            n_layers_surv_piecewise=n_layers)
                model_hivae_2 = model_loading(input_dim=data.shape[1],
                            z_dim=params["z_dim"],
                            y_dim=params["y_dim"],
                            s_dim=params["s_dim"],
                            y_dim_partition=None,
                            feat_types_dict=feat_types_dict,
                            intervals_surv_piecewise=intervals,
                            n_layers_surv_piecewise=n_layers)
                model_hivae_1, _, _ = train_HIVAE(model_hivae_1, train_data, train_miss_mask, train_true_miss_mask, feat_types_dict, batch_size, params["lr"], epochs, seed=seed)
                model_hivae_2, _, _ = train_HIVAE(model_hivae_2, train_data, train_miss_mask, train_true_miss_mask, feat_types_dict, batch_size, params["lr"], epochs, seed=seed+1)
                # Generate
                if condition is not None:
                    raise NotImplementedError("Condition not implemented for this method")
                else:
                    est_data_gen_transformed_1 = generate_from_HIVAE(model_hivae_1, test_data, test_miss_mask, test_true_miss_mask,
                                                                feat_types_dict, n_generated_dataset=n_generated_dataset//2, n_generated_sample=test_data.shape[0], from_prior=gen_from_prior)
                    est_data_gen_transformed_2 = generate_from_HIVAE(model_hivae_2, test_data, test_miss_mask, test_true_miss_mask,
                                                                feat_types_dict, n_generated_dataset=n_generated_dataset - n_generated_dataset//2, n_generated_sample=test_data.shape[0], from_prior=gen_from_prior)
                    est_data_gen_transformed = torch.cat([est_data_gen_transformed_1, est_data_gen_transformed_2], dim=0)
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
            
                # # k-fold cross-validation
                # kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                # for train_index, test_index in kf.split(data):
                #     train_data, test_data = data[train_index], data[test_index]
                #     df_test_data = df.iloc[test_index]
                #     test_data_loader = SurvivalAnalysisDataLoader(df_test_data, target_column = "censor", time_to_event_column = "time")
                #     full_data_loader = SurvivalAnalysisDataLoader(df, target_column = "censor", time_to_event_column = "time")
                #     train_miss_mask, test_miss_mask = miss_mask[train_index], miss_mask[test_index]
                #     train_true_miss_mask, test_true_miss_mask = true_miss_mask[train_index], true_miss_mask[test_index]
                    
                #     if method == 'train_train_gen_full':
                #         # Train
                #         batch_size = params["batch_size"]
                #         batch_size = min(batch_size, data.shape[0])
                #         model_hivae = model_loading(input_dim=data.shape[1],
                #                     z_dim=params["z_dim"],
                #                     y_dim=params["y_dim"],
                #                     s_dim=params["s_dim"],
                #                     y_dim_partition=None,
                #                     feat_types_dict=feat_types_dict,
                #                     intervals_surv_piecewise=intervals,
                #                     n_layers_surv_piecewise=n_layers)
                #         model_hivae, _, _ = train_HIVAE(model_hivae, train_data, train_miss_mask, train_true_miss_mask, feat_types_dict, batch_size, params["lr"], epochs)
                #         # Generate
                #         if condition is not None:
                #             est_data_gen_transformed = generate_from_condition_HIVAE(model_hivae, df, miss_mask, true_miss_mask,
                #                                                                     feat_types_dict, n_generated_dataset, n_generated_sample=data.shape[0], from_prior=gen_from_prior, condition=condition)
                #         else:
                #             est_data_gen_transformed = generate_from_HIVAE(model_hivae, data, miss_mask, true_miss_mask,
                #                                                         feat_types_dict, n_generated_dataset=n_generated_dataset, n_generated_sample=data.shape[0], from_prior=gen_from_prior)
                #         score_k = []
                #         for j in range(n_generated_dataset):
                #             df_gen_data = pd.DataFrame(est_data_gen_transformed[j].numpy(), columns=columns)
                #             if metric == 'log_rank_test':
                #                 score_kj = metrics.compute_logrank_test(df, df_gen_data)
                #             else: # 'survival_km_distance'
                #                 gen_data = SurvivalAnalysisDataLoader(df_gen_data, target_column = "censor", time_to_event_column = "time")
                #                 clear_cache()
                #                 evaluation = Metrics().evaluate(X_gt=full_data_loader, # can be dataloaders or dataframes
                #                                                 X_syn=gen_data, 
                #                                                 reduction='mean', # default mean
                #                                                 n_histogram_bins=10, # default 10
                #                                                 n_folds=1,
                #                                                 metrics={'stats': ['survival_km_distance']},
                #                                                 task_type='survival_analysis', 
                #                                                 use_cache=True)
                #                 score_kj = evaluation.T[["stats.survival_km_distance.abs_optimism"]].T["mean"].values[0]
                #             score_k.append(score_kj)

                #     else:
                #         # Train
                #         batch_size = params["batch_size"]
                #         batch_size = min(batch_size, data.shape[0])
                #         model_hivae = model_loading(input_dim=data.shape[1],
                #                     z_dim=params["z_dim"],
                #                     y_dim=params["y_dim"],
                #                     s_dim=params["s_dim"],
                #                     y_dim_partition=None,
                #                     feat_types_dict=feat_types_dict,
                #                     intervals_surv_piecewise=intervals,
                #                     n_layers_surv_piecewise=n_layers)
                #         model_hivae, _, _ = train_HIVAE(model_hivae, train_data, train_miss_mask, train_true_miss_mask, feat_types_dict, batch_size, params["lr"], epochs)
                #         # Generate
                #         est_data_gen_transformed = generate_from_HIVAE(model_hivae, test_data, test_miss_mask, test_true_miss_mask,
                #                                                         feat_types_dict, n_generated_dataset=n_generated_dataset, n_generated_sample=test_data.shape[0], from_prior=gen_from_prior)

                #         score_k = []
                #         for j in range(n_generated_dataset):
                #             df_gen_data = pd.DataFrame(est_data_gen_transformed[j].numpy(), columns=columns)
                #             if metric == 'log_rank_test':
                #                 score_kj = metrics.compute_logrank_test(df_test_data, df_gen_data)
                #             else: # 'survival_km_distance'
                #                 gen_data = SurvivalAnalysisDataLoader(df_gen_data, target_column = "censor", time_to_event_column = "time")
                #                 clear_cache()
                #                 evaluation = Metrics().evaluate(X_gt=test_data_loader, # can be dataloaders or dataframes
                #                                                 X_syn=gen_data, 
                #                                                 reduction='mean', # default mean
                #                                                 n_histogram_bins=10, # default 10
                #                                                 n_folds=1,
                #                                                 metrics={'stats': ['survival_km_distance']},
                #                                                 task_type='survival_analysis', 
                #                                                 use_cache=True)
                #                 score_kj = evaluation.T[["stats.survival_km_distance.abs_optimism"]].T["mean"].values[0]
                #             score_k.append(score_kj)
                #     scores.append(np.mean(score_k))
            print(f"Score: {np.mean(scores)}")
        except Exception as e:  # invalid set of params
            print(f"{type(e).__name__}: {e}")
            print(params)
            raise optuna.TrialPruned()
        return np.mean(scores)
    

    db_file = study_name + '.db'

    full_optuna_study_path = study_path + db_file
    if os.path.exists(full_optuna_study_path):
        print("This optuna study ({}) already exists. We load the study from the existing file.".format(db_file))
        study = optuna.load_study(study_name=study_name, storage='sqlite:///'+full_optuna_study_path)

    else: 
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="minimize", study_name=study_name, storage='sqlite:///'+full_optuna_study_path, sampler=sampler)
        if "HI-VAE_piecewise" in generator_name:
            default_params = {"lr": 1e-3, "batch_size": 100, "z_dim": 20, "y_dim": 15, "s_dim": 20, "n_layers_surv_piecewise": 1, "n_intervals": 10}
        else:
            default_params = {"lr": 1e-3, "batch_size": 100, "z_dim": 20, "y_dim": 15, "s_dim": 20}
        study.enqueue_trial(default_params)
        print("Enqueued trial:", study.get_trials(deepcopy=False))
    study.optimize(objective, n_trials=n_trials)
    study.best_params

    return study.best_params, study


def optuna_hyperparameter_search_multisim(df_list, miss_mask_list, true_miss_mask_list, feat_types_dict,
                                          n_generated_dataset, n_splits, n_trials, columns, generator_name,
                                          epochs=1000, n_generated_sample=None,
                                          study_name='optuna_study_surv_hivae_multisim', study_path='',
                                          metric='survival_km_distance', method='',
                                          gen_from_prior=False, condition=None, cond_df_list=None,
                                          batchcorrect=False, seed=1,
                                          screening_epochs=251, n_startup_trials=20,
                                          target_epsilon=None, target_delta=1e-5, max_grad_norm=1.0):
    if "_DP" in generator_name and target_epsilon is None:
        raise ValueError("target_epsilon must be set when generator_name contains '_DP'.")

    model_name = "HIVAE_inputDropout"
    n_simulation_seeds = len(df_list)
    if len(miss_mask_list) != n_simulation_seeds or len(true_miss_mask_list) != n_simulation_seeds:
        raise ValueError("miss_mask_list and true_miss_mask_list must have the same length as df_list")
    if condition is not None:
        if cond_df_list is None or len(cond_df_list) != n_simulation_seeds:
            raise ValueError("cond_df_list must be provided and have the same length as df_list when condition is set")
        cond_loaders = [SurvivalAnalysisDataLoader(cdf, target_column="censor", time_to_event_column="time")
                        for cdf in cond_df_list]
    else:
        cond_loaders = None

    def _train_chunk(model, data, miss_mask, true_miss_mask, batch_size, lr, n_epochs, seed_, start_epoch=0):
        # Normalize the return to a 3-tuple regardless of training routine,
        # so the downstream unpackers `models[i], _, loss_val_i = ...` work
        # the same way for DP and non-DP paths. The DP variant additionally
        # returns final_epsilon; discard it here (train_HIVAE_DP prints it
        # to stdout when verbose=True).
        if "_DP" in generator_name:
            model_, lt, lv, _ = train_HIVAE_DP(
                model, data, miss_mask, true_miss_mask, feat_types_dict,
                batch_size, lr, n_epochs,
                target_epsilon=target_epsilon, target_delta=target_delta, max_grad_norm=max_grad_norm,
                seed=seed_, start_epoch=start_epoch,
            )
            return model_, lt, lv
        if batchcorrect:
            return train_HIVAE_bis(model, data, miss_mask, true_miss_mask, feat_types_dict,
                                   batch_size, lr, n_epochs, seed=seed_, start_epoch=start_epoch)
        return train_HIVAE(model, data, miss_mask, true_miss_mask, feat_types_dict,
                           batch_size, lr, n_epochs, seed=seed_, start_epoch=start_epoch)

    def _build_model(input_dim, params, intervals_i, n_layers):
        model_loading = getattr(importlib.import_module("src"), model_name)
        return model_loading(input_dim=input_dim,
                             z_dim=params["z_dim"],
                             y_dim=params["y_dim"],
                             s_dim=params["s_dim"],
                             y_dim_partition=None,
                             feat_types_dict=feat_types_dict,
                             intervals_surv_piecewise=intervals_i,
                             n_layers_surv_piecewise=n_layers)

    def _evaluate_full_or_train_full(model, data_i, miss_i, true_miss_i, df_i, gt_loader, n_gen_sample_i):
        if condition is not None:
            est = generate_from_condition_HIVAE(model, df_i, miss_i, true_miss_i, feat_types_dict,
                                                n_generated_dataset, n_generated_sample=data_i.shape[0],
                                                from_prior=gen_from_prior, condition=condition)
        else:
            est = generate_from_HIVAE(model, data_i, miss_i, true_miss_i, feat_types_dict,
                                      n_generated_dataset, n_generated_sample=n_gen_sample_i,
                                      from_prior=gen_from_prior)
        full_data_tensor = torch.cat(list(est), dim=0)
        df_gen = pd.DataFrame(full_data_tensor.numpy(), columns=columns)
        df_gen["time"] = df_gen["time"].clip(lower=1e-6)
        gen_data = SurvivalAnalysisDataLoader(df_gen, target_column="censor", time_to_event_column="time")
        clear_cache()
        evaluation = Metrics().evaluate(X_gt=gt_loader, X_syn=gen_data, reduction='mean',
                                        n_histogram_bins=10, n_folds=1,
                                        metrics={'stats': ['survival_km_distance']},
                                        task_type='survival_analysis', use_cache=True)
        return evaluation.T[["stats.survival_km_distance.abs_optimism"]].T["mean"].values[0]

    def objective(trial: optuna.Trial):
        set_seed()
        hp_space = hyperparameter_space(df_list[0], n_splits, generator_name)
        params = suggest_all(trial, hp_space)
        if "HI-VAE_piecewise" in generator_name:
            intervals_list = [get_intervals(df, params["n_intervals"]) for df in df_list]
            n_layers = params["n_layers_surv_piecewise"]
        else:
            intervals_list = [None] * n_simulation_seeds
            n_layers = None
        print(f"trial_{trial.number}")
        print(f"Hyperparameters: {params}")

        data_list = [torch.from_numpy(df.values) for df in df_list]
        scores = []
        try:
            if method == 'train_full_gen_full':
                full_loaders = [SurvivalAnalysisDataLoader(df, target_column="censor", time_to_event_column="time")
                                for df in df_list]
                models, batch_sizes = [], []
                for i in range(n_simulation_seeds):
                    bs = min(params["batch_size"], int(0.9 * data_list[i].shape[0]))
                    batch_sizes.append(bs)
                    models.append(_build_model(data_list[i].shape[1], params, intervals_list[i], n_layers))

                screening_val_losses = []
                for i in range(n_simulation_seeds):
                    np.random.seed(seed)
                    models[i], _, loss_val_i = _train_chunk(models[i], data_list[i], miss_mask_list[i],
                                                            true_miss_mask_list[i], batch_sizes[i],
                                                            params["lr"], screening_epochs, seed)
                    screening_val_losses.append(float(loss_val_i[-1]))
                avg_val = float(np.mean(screening_val_losses))
                print(f"Screening avg val loss @ epoch {screening_epochs}: {avg_val}")
                trial.report(avg_val, step=screening_epochs)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                remaining_epochs = max(0, epochs - screening_epochs)
                if remaining_epochs > 0:
                    for i in range(n_simulation_seeds):
                        np.random.seed(seed)
                        models[i], _, _ = _train_chunk(models[i], data_list[i], miss_mask_list[i],
                                                       true_miss_mask_list[i], batch_sizes[i],
                                                       params["lr"], remaining_epochs, seed,
                                                       start_epoch=screening_epochs)

                for i in range(n_simulation_seeds):
                    n_gen_sample_i = n_generated_sample if n_generated_sample is not None else data_list[i].shape[0]
                    gt_loader = cond_loaders[i] if condition is not None else full_loaders[i]
                    scores.append(_evaluate_full_or_train_full(
                        models[i], data_list[i], miss_mask_list[i], true_miss_mask_list[i],
                        df_list[i], gt_loader, n_gen_sample_i))

            elif method == 'train_train_gen_full':
                full_loaders = [SurvivalAnalysisDataLoader(df, target_column="censor", time_to_event_column="time")
                                for df in df_list]
                train_test_share = .8
                train_data_list, train_miss_list, train_true_miss_list = [], [], []
                models, batch_sizes = [], []
                for i in range(n_simulation_seeds):
                    n_samples_i = data_list[i].shape[0]
                    n_train_i = int(train_test_share * n_samples_i)
                    train_idx = np.random.choice(n_samples_i, n_train_i, replace=False)
                    train_data_list.append(data_list[i][train_idx])
                    train_miss_list.append(miss_mask_list[i][train_idx])
                    train_true_miss_list.append(true_miss_mask_list[i][train_idx])
                    batch_sizes.append(min(params["batch_size"], train_data_list[i].shape[0]))
                    models.append(_build_model(data_list[i].shape[1], params, intervals_list[i], n_layers))

                screening_val_losses = []
                for i in range(n_simulation_seeds):
                    np.random.seed(seed)
                    models[i], _, loss_val_i = _train_chunk(models[i], train_data_list[i], train_miss_list[i],
                                                            train_true_miss_list[i], batch_sizes[i],
                                                            params["lr"], screening_epochs, seed)
                    screening_val_losses.append(float(loss_val_i[-1]))
                avg_val = float(np.mean(screening_val_losses))
                print(f"Screening avg val loss @ epoch {screening_epochs}: {avg_val}")
                trial.report(avg_val, step=screening_epochs)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                remaining_epochs = max(0, epochs - screening_epochs)
                if remaining_epochs > 0:
                    for i in range(n_simulation_seeds):
                        np.random.seed(seed)
                        models[i], _, _ = _train_chunk(models[i], train_data_list[i], train_miss_list[i],
                                                       train_true_miss_list[i], batch_sizes[i],
                                                       params["lr"], remaining_epochs, seed,
                                                       start_epoch=screening_epochs)

                for i in range(n_simulation_seeds):
                    n_gen_sample_i = n_generated_sample if n_generated_sample is not None else data_list[i].shape[0]
                    gt_loader = cond_loaders[i] if condition is not None else full_loaders[i]
                    scores.append(_evaluate_full_or_train_full(
                        models[i], data_list[i], miss_mask_list[i], true_miss_mask_list[i],
                        df_list[i], gt_loader, n_gen_sample_i))

            elif method == 'train_train_gen_test':
                if condition is not None:
                    raise NotImplementedError("Condition not implemented for this method")
                train_test_share = .8
                train_data_list, train_miss_list, train_true_miss_list = [], [], []
                test_data_list, test_miss_list, test_true_miss_list = [], [], []
                test_loaders, models, batch_sizes = [], [], []
                for i in range(n_simulation_seeds):
                    n_samples_i = data_list[i].shape[0]
                    n_train_i = int(train_test_share * n_samples_i)
                    train_idx = np.random.choice(n_samples_i, n_train_i, replace=False)
                    test_idx = [j for j in np.arange(n_samples_i) if j not in train_idx]
                    train_data_list.append(data_list[i][train_idx])
                    train_miss_list.append(miss_mask_list[i][train_idx])
                    train_true_miss_list.append(true_miss_mask_list[i][train_idx])
                    test_data_list.append(data_list[i][test_idx])
                    test_miss_list.append(miss_mask_list[i][test_idx])
                    test_true_miss_list.append(true_miss_mask_list[i][test_idx])
                    test_loaders.append(SurvivalAnalysisDataLoader(df_list[i].iloc[test_idx],
                                                                   target_column="censor",
                                                                   time_to_event_column="time"))
                    batch_sizes.append(min(params["batch_size"], train_data_list[i].shape[0]))
                    models.append(_build_model(data_list[i].shape[1], params, intervals_list[i], n_layers))

                screening_val_losses = []
                for i in range(n_simulation_seeds):
                    np.random.seed(seed)
                    models[i], _, loss_val_i = _train_chunk(models[i], train_data_list[i], train_miss_list[i],
                                                            train_true_miss_list[i], batch_sizes[i],
                                                            params["lr"], screening_epochs, seed)
                    screening_val_losses.append(float(loss_val_i[-1]))
                avg_val = float(np.mean(screening_val_losses))
                print(f"Screening avg val loss @ epoch {screening_epochs}: {avg_val}")
                trial.report(avg_val, step=screening_epochs)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                remaining_epochs = max(0, epochs - screening_epochs)
                if remaining_epochs > 0:
                    for i in range(n_simulation_seeds):
                        np.random.seed(seed)
                        models[i], _, _ = _train_chunk(models[i], train_data_list[i], train_miss_list[i],
                                                       train_true_miss_list[i], batch_sizes[i],
                                                       params["lr"], remaining_epochs, seed,
                                                       start_epoch=screening_epochs)

                for i in range(n_simulation_seeds):
                    est = generate_from_HIVAE(models[i], test_data_list[i], test_miss_list[i],
                                              test_true_miss_list[i], feat_types_dict,
                                              n_generated_dataset, n_generated_sample=test_data_list[i].shape[0],
                                              from_prior=gen_from_prior)
                    full_data_tensor = torch.cat(list(est), dim=0)
                    df_gen = pd.DataFrame(full_data_tensor.numpy(), columns=columns)
                    df_gen["time"] = df_gen["time"].clip(lower=1e-6)
                    gen_data = SurvivalAnalysisDataLoader(df_gen, target_column="censor", time_to_event_column="time")
                    clear_cache()
                    evaluation = Metrics().evaluate(X_gt=test_loaders[i], X_syn=gen_data, reduction='mean',
                                                    n_histogram_bins=10, n_folds=1,
                                                    metrics={'stats': ['survival_km_distance']},
                                                    task_type='survival_analysis', use_cache=True)
                    scores.append(evaluation.T[["stats.survival_km_distance.abs_optimism"]].T["mean"].values[0])

            else:
                raise ValueError("Invalid method")

            print(f"Score: {np.mean(scores)}")
        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"{type(e).__name__}: {e}")
            print(params)
            raise optuna.TrialPruned()
        return float(np.mean(scores))

    db_file = study_name + '.db'
    full_optuna_study_path = study_path + db_file
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=n_startup_trials)
    if os.path.exists(full_optuna_study_path):
        print("This optuna study ({}) already exists. We load the study from the existing file.".format(db_file))
        study = optuna.load_study(study_name=study_name, storage='sqlite:///'+full_optuna_study_path,
                                  sampler=sampler, pruner=pruner)
    else:
        study = optuna.create_study(direction="minimize", study_name=study_name,
                                    storage='sqlite:///'+full_optuna_study_path,
                                    sampler=sampler, pruner=pruner)
        if "HI-VAE_piecewise" in generator_name:
            default_params = {"lr": 1e-3, "batch_size": 100, "z_dim": 20, "y_dim": 15, "s_dim": 20,
                              "n_layers_surv_piecewise": 1, "n_intervals": 10}
        else:
            default_params = {"lr": 1e-3, "batch_size": 100, "z_dim": 20, "y_dim": 15, "s_dim": 20}
        study.enqueue_trial(default_params)
        print("Enqueued trial:", study.get_trials(deepcopy=False))

    study.optimize(objective, n_trials=n_trials)

    return study.best_params, study


def optuna_hyperparameter_search_multiseed(df, miss_mask, true_miss_mask, feat_types_dict,
                                           n_generated_dataset, n_splits, n_trials, columns, generator_name,
                                           epochs=1000, n_generated_sample=None,
                                           study_name='optuna_study_surv_hivae_multiseed', study_path='',
                                           metric='survival_km_distance', method='',
                                           gen_from_prior=False, condition=None, cond_df=None,
                                           batchcorrect=False, seed=10,
                                           n_training_seeds=3,
                                           screening_epochs=251, n_startup_trials=20,
                                           target_epsilon=None, target_delta=1e-5, max_grad_norm=1.0):
    if "_DP" in generator_name and target_epsilon is None:
        raise ValueError("target_epsilon must be set when generator_name contains '_DP'.")

    model_name = "HIVAE_inputDropout"
    if condition is not None and cond_df is None:
        raise ValueError("cond_df must be provided when condition is not None")
    cond_loader = (SurvivalAnalysisDataLoader(cond_df, target_column="censor", time_to_event_column="time")
                   if condition is not None else None)

    def _train_chunk(model, data, miss_mask_, true_miss_mask_, batch_size, lr, n_epochs, seed_, start_epoch=0):
        # Normalize the return to a 3-tuple regardless of training routine,
        # so the downstream unpackers `models[i], _, loss_val_i = ...` work
        # the same way for DP and non-DP paths. The DP variant additionally
        # returns final_epsilon; discard it here (train_HIVAE_DP prints it
        # to stdout when verbose=True).
        if "_DP" in generator_name:
            model_, lt, lv, _ = train_HIVAE_DP(
                model, data, miss_mask_, true_miss_mask_, feat_types_dict,
                batch_size, lr, n_epochs,
                target_epsilon=target_epsilon, target_delta=target_delta, max_grad_norm=max_grad_norm,
                seed=seed_, start_epoch=start_epoch,
            )
            return model_, lt, lv
        if batchcorrect:
            return train_HIVAE_bis(model, data, miss_mask_, true_miss_mask_, feat_types_dict,
                                   batch_size, lr, n_epochs, seed=seed_, start_epoch=start_epoch)
        return train_HIVAE(model, data, miss_mask_, true_miss_mask_, feat_types_dict,
                           batch_size, lr, n_epochs, seed=seed_, start_epoch=start_epoch)

    def _build_model(input_dim, params, intervals_, n_layers):
        model_loading = getattr(importlib.import_module("src"), model_name)
        return model_loading(input_dim=input_dim,
                             z_dim=params["z_dim"],
                             y_dim=params["y_dim"],
                             s_dim=params["s_dim"],
                             y_dim_partition=None,
                             feat_types_dict=feat_types_dict,
                             intervals_surv_piecewise=intervals_,
                             n_layers_surv_piecewise=n_layers)

    def _evaluate_full_or_train_full(model, data_for_gen, miss_for_gen, true_miss_for_gen,
                                     df_for_cond, gt_loader, n_gen_sample_):
        if condition is not None:
            est = generate_from_condition_HIVAE(model, df_for_cond, miss_for_gen, true_miss_for_gen, feat_types_dict,
                                                n_generated_dataset, n_generated_sample=data_for_gen.shape[0],
                                                from_prior=gen_from_prior, condition=condition)
        else:
            est = generate_from_HIVAE(model, data_for_gen, miss_for_gen, true_miss_for_gen, feat_types_dict,
                                      n_generated_dataset, n_generated_sample=n_gen_sample_,
                                      from_prior=gen_from_prior)
        full_data_tensor = torch.cat(list(est), dim=0)
        df_gen = pd.DataFrame(full_data_tensor.numpy(), columns=columns)
        df_gen["time"] = df_gen["time"].clip(lower=1e-6)
        gen_data = SurvivalAnalysisDataLoader(df_gen, target_column="censor", time_to_event_column="time")
        clear_cache()
        evaluation = Metrics().evaluate(X_gt=gt_loader, X_syn=gen_data, reduction='mean',
                                        n_histogram_bins=10, n_folds=1,
                                        metrics={'stats': ['survival_km_distance']},
                                        task_type='survival_analysis', use_cache=True)
        return evaluation.T[["stats.survival_km_distance.abs_optimism"]].T["mean"].values[0]

    def objective(trial: optuna.Trial):
        set_seed()
        hp_space = hyperparameter_space(df, n_splits, generator_name)
        params = suggest_all(trial, hp_space)
        if "HI-VAE_piecewise" in generator_name:
            intervals = get_intervals(df, params["n_intervals"])
            n_layers = params["n_layers_surv_piecewise"]
        else:
            intervals = None
            n_layers = None
        print(f"trial_{trial.number}")
        print(f"Hyperparameters: {params}")

        data = torch.from_numpy(df.values)
        scores = []
        try:
            if method == 'train_full_gen_full':
                full_loader = SurvivalAnalysisDataLoader(df, target_column="censor", time_to_event_column="time")
                batch_size = min(params["batch_size"], int(0.9 * data.shape[0]))
                models = [_build_model(data.shape[1], params, intervals, n_layers)
                          for _ in range(n_training_seeds)]

                screening_val_losses = []
                for i in range(n_training_seeds):
                    np.random.seed(seed + i)
                    models[i], _, loss_val_i = _train_chunk(models[i], data, miss_mask, true_miss_mask,
                                                            batch_size, params["lr"], screening_epochs, seed + i)
                    screening_val_losses.append(float(loss_val_i[-1]))
                avg_val = float(np.mean(screening_val_losses))
                print(f"Screening avg val loss @ epoch {screening_epochs}: {avg_val}")
                trial.report(avg_val, step=screening_epochs)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                remaining_epochs = max(0, epochs - screening_epochs)
                if remaining_epochs > 0:
                    for i in range(n_training_seeds):
                        np.random.seed(seed + i)
                        models[i], _, _ = _train_chunk(models[i], data, miss_mask, true_miss_mask,
                                                       batch_size, params["lr"], remaining_epochs, seed + i,
                                                       start_epoch=screening_epochs)

                gt_loader = cond_loader if condition is not None else full_loader
                n_gen_sample_ = n_generated_sample if n_generated_sample is not None else data.shape[0]
                for i in range(n_training_seeds):
                    scores.append(_evaluate_full_or_train_full(
                        models[i], data, miss_mask, true_miss_mask, df, gt_loader, n_gen_sample_))

            elif method == 'train_train_gen_full':
                full_loader = SurvivalAnalysisDataLoader(df, target_column="censor", time_to_event_column="time")
                train_test_share = .8
                train_data_list, train_miss_list, train_true_miss_list = [], [], []
                models, batch_sizes = [], []
                for i in range(n_training_seeds):
                    n_samples_ = data.shape[0]
                    n_train_ = int(train_test_share * n_samples_)
                    train_idx = np.random.choice(n_samples_, n_train_, replace=False)
                    train_data_list.append(data[train_idx])
                    train_miss_list.append(miss_mask[train_idx])
                    train_true_miss_list.append(true_miss_mask[train_idx])
                    batch_sizes.append(min(params["batch_size"], train_data_list[i].shape[0]))
                    models.append(_build_model(data.shape[1], params, intervals, n_layers))

                screening_val_losses = []
                for i in range(n_training_seeds):
                    np.random.seed(seed + i)
                    models[i], _, loss_val_i = _train_chunk(models[i], train_data_list[i], train_miss_list[i],
                                                            train_true_miss_list[i], batch_sizes[i],
                                                            params["lr"], screening_epochs, seed + i)
                    screening_val_losses.append(float(loss_val_i[-1]))
                avg_val = float(np.mean(screening_val_losses))
                print(f"Screening avg val loss @ epoch {screening_epochs}: {avg_val}")
                trial.report(avg_val, step=screening_epochs)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                remaining_epochs = max(0, epochs - screening_epochs)
                if remaining_epochs > 0:
                    for i in range(n_training_seeds):
                        np.random.seed(seed + i)
                        models[i], _, _ = _train_chunk(models[i], train_data_list[i], train_miss_list[i],
                                                       train_true_miss_list[i], batch_sizes[i],
                                                       params["lr"], remaining_epochs, seed + i,
                                                       start_epoch=screening_epochs)

                gt_loader = cond_loader if condition is not None else full_loader
                n_gen_sample_ = n_generated_sample if n_generated_sample is not None else data.shape[0]
                for i in range(n_training_seeds):
                    scores.append(_evaluate_full_or_train_full(
                        models[i], data, miss_mask, true_miss_mask, df, gt_loader, n_gen_sample_))

            elif method == 'train_train_gen_test':
                if condition is not None:
                    raise NotImplementedError("Condition not implemented for this method")
                train_test_share = .8
                train_data_list, train_miss_list, train_true_miss_list = [], [], []
                test_data_list, test_miss_list, test_true_miss_list = [], [], []
                test_loaders, models, batch_sizes = [], [], []
                for i in range(n_training_seeds):
                    n_samples_ = data.shape[0]
                    n_train_ = int(train_test_share * n_samples_)
                    train_idx = np.random.choice(n_samples_, n_train_, replace=False)
                    test_idx = [j for j in np.arange(n_samples_) if j not in train_idx]
                    train_data_list.append(data[train_idx])
                    train_miss_list.append(miss_mask[train_idx])
                    train_true_miss_list.append(true_miss_mask[train_idx])
                    test_data_list.append(data[test_idx])
                    test_miss_list.append(miss_mask[test_idx])
                    test_true_miss_list.append(true_miss_mask[test_idx])
                    test_loaders.append(SurvivalAnalysisDataLoader(df.iloc[test_idx],
                                                                   target_column="censor",
                                                                   time_to_event_column="time"))
                    batch_sizes.append(min(params["batch_size"], train_data_list[i].shape[0]))
                    models.append(_build_model(data.shape[1], params, intervals, n_layers))

                screening_val_losses = []
                for i in range(n_training_seeds):
                    np.random.seed(seed + i)
                    models[i], _, loss_val_i = _train_chunk(models[i], train_data_list[i], train_miss_list[i],
                                                            train_true_miss_list[i], batch_sizes[i],
                                                            params["lr"], screening_epochs, seed + i)
                    screening_val_losses.append(float(loss_val_i[-1]))
                avg_val = float(np.mean(screening_val_losses))
                print(f"Screening avg val loss @ epoch {screening_epochs}: {avg_val}")
                trial.report(avg_val, step=screening_epochs)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                remaining_epochs = max(0, epochs - screening_epochs)
                if remaining_epochs > 0:
                    for i in range(n_training_seeds):
                        np.random.seed(seed + i)
                        models[i], _, _ = _train_chunk(models[i], train_data_list[i], train_miss_list[i],
                                                       train_true_miss_list[i], batch_sizes[i],
                                                       params["lr"], remaining_epochs, seed + i,
                                                       start_epoch=screening_epochs)

                for i in range(n_training_seeds):
                    est = generate_from_HIVAE(models[i], test_data_list[i], test_miss_list[i],
                                              test_true_miss_list[i], feat_types_dict,
                                              n_generated_dataset, n_generated_sample=test_data_list[i].shape[0],
                                              from_prior=gen_from_prior)
                    full_data_tensor = torch.cat(list(est), dim=0)
                    df_gen = pd.DataFrame(full_data_tensor.numpy(), columns=columns)
                    df_gen["time"] = df_gen["time"].clip(lower=1e-6)
                    gen_data = SurvivalAnalysisDataLoader(df_gen, target_column="censor", time_to_event_column="time")
                    clear_cache()
                    evaluation = Metrics().evaluate(X_gt=test_loaders[i], X_syn=gen_data, reduction='mean',
                                                    n_histogram_bins=10, n_folds=1,
                                                    metrics={'stats': ['survival_km_distance']},
                                                    task_type='survival_analysis', use_cache=True)
                    scores.append(evaluation.T[["stats.survival_km_distance.abs_optimism"]].T["mean"].values[0])

            else:
                raise ValueError("Invalid method")

            print(f"Score: {np.mean(scores)}")
        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"{type(e).__name__}: {e}")
            print(params)
            raise optuna.TrialPruned()
        return float(np.mean(scores))

    db_file = study_name + '.db'
    full_optuna_study_path = study_path + db_file
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=n_startup_trials)
    if os.path.exists(full_optuna_study_path):
        print("This optuna study ({}) already exists. We load the study from the existing file.".format(db_file))
        study = optuna.load_study(study_name=study_name, storage='sqlite:///'+full_optuna_study_path,
                                  sampler=sampler, pruner=pruner)
    else:
        study = optuna.create_study(direction="minimize", study_name=study_name,
                                    storage='sqlite:///'+full_optuna_study_path,
                                    sampler=sampler, pruner=pruner)
        if "HI-VAE_piecewise" in generator_name:
            default_params = {"lr": 1e-3, "batch_size": 100, "z_dim": 20, "y_dim": 15, "s_dim": 20,
                              "n_layers_surv_piecewise": 1, "n_intervals": 10}
        else:
            default_params = {"lr": 1e-3, "batch_size": 100, "z_dim": 20, "y_dim": 15, "s_dim": 20}
        study.enqueue_trial(default_params)
        print("Enqueued trial:", study.get_trials(deepcopy=False))

    study.optimize(objective, n_trials=n_trials)

    return study.best_params, study
