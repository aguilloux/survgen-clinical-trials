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

def train_HIVAE(vae_model, data, miss_mask, true_miss_mask, feat_types_dict, batch_size, lr, epochs, verbose = True):

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

    # Training
    optimizer = optim.Adam(vae_model.parameters(), lr=lr)

    start_time = time.time()
    loss_train, error_observed_train, error_missing_train = [], [], []
    loss_val, error_observed_val, error_missing_val = [], [], []

    rng = np.random.default_rng(seed=42)
    # Setting for early stopping
    best_val_loss = float('inf')

    patience = 10 # 5
    n_iter_validation = 50
    n_iter_min = 100
    counter = 0
    # min_improvement_ratio = 0 #5e-3
    for epoch in range(epochs):
        avg_loss, avg_KL_s, avg_KL_z = 0.0, 0.0, 0.0
        avg_loss_val, avg_KL_s_val, avg_KL_z_val = 0.0, 0.0, 0.0
        samples_list, p_params_list, q_params_list, log_p_x_total, log_p_x_missing_total = [], [], [], [], []
        tau = max(1.0 - 0.01 * epoch, 1e-3)

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
                                                                                    feat_types_dict, miss_mask[:n_train_samples])
        
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
            if epoch % 100 == 0:
                visualization.print_loss(epoch, start_time, -avg_loss, avg_KL_s, avg_KL_z)
        

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
                print(f"Early stopping at epoch {epoch}.")
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
                _, normalization_params = data_processing.batch_normalization(data_list_observed, vae_model.feat_types_list, miss_list)

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
            _, normalization_params = data_processing.batch_normalization(data_list_observed, feat_types_dict, miss_list)

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



def run(df, miss_mask, true_miss_mask, feat_types_dict,  n_generated_dataset, n_generated_sample=None,
        params={"lr": 1e-3, "batch_size": 100, "z_dim": 20, "y_dim": 15, "s_dim": 20, "n_layers_surv_piecewise": 1, "n_intervals": 10}, 
        epochs=1000, verbose=True, plot=False, gen_from_prior=False, condition=None, differential_privacy=False, batchcorrect=False):

    set_seed()
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
        model_hivae, loss_train, loss_val = train_HIVAE_DP(model_hivae, data, miss_mask, true_miss_mask, feat_types_dict, batch_size, lr, epochs, verbose)
    else:
        if batchcorrect:
            model_hivae, loss_train, loss_val = train_HIVAE_bis(model_hivae, data, miss_mask, true_miss_mask, feat_types_dict, batch_size, lr, epochs, verbose)
        else:
            model_hivae, loss_train, loss_val = train_HIVAE(model_hivae, data, miss_mask, true_miss_mask, feat_types_dict, batch_size, lr, epochs, verbose)
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
        CategoricalDistribution(name="lr", choices=[1e-3, 2e-4, 1e-4]),
        CategoricalDistribution(name="batch_size", choices=get_batchsize(n_samples, n_splits) + [100]),
        IntegerDistribution(name="z_dim", low=10, high=200, step=10),
        IntegerDistribution(name="y_dim", low=10, high=200, step=5),
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
    batch_size_ratio = [.25, .4, .6, .75]
    batch_size = [int(ratio * n_samples * (n_splits - 1) / n_splits) for ratio in batch_size_ratio]

    return batch_size

def optuna_hyperparameter_search(df, miss_mask, true_miss_mask, feat_types_dict, n_generated_dataset, n_splits, n_trials, columns, generator_name, epochs = 1000, n_generated_sample = None, study_name='optuna_study_surv_hivae', metric='survival_km_distance', method='', gen_from_prior=False, condition=None, cond_df=None, batchcorrect=False, seed=10):
   
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
                    model_hivae, _, _ = train_HIVAE_DP(model_hivae, data, miss_mask, true_miss_mask, feat_types_dict, batch_size, params["lr"], epochs)
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

def train_HIVAE_DP(vae_model, data, miss_mask, true_miss_mask, feat_types_dict, batch_size, lr, epochs, verbose = True):

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

    # Training
    optimizer = optim.Adam(vae_model.parameters(), lr=lr)

    start_time = time.time()
    loss_train, error_observed_train, error_missing_train = [], [], []
    loss_val, error_observed_val, error_missing_val = [], [], []

    rng = np.random.default_rng(seed=42)
    # Setting for early stopping
    best_val_loss = float('inf')

    patience = 10 # 5
    n_iter_validation = 50
    n_iter_min = 100
    counter = 0

    privacy_engine = PrivacyEngine()
    dataset = MyCustomDataset(data_train, miss_mask_train, feat_types_dict)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) #, drop_last=True)
    # vae_model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    #     module=vae_model,
    #     optimizer=optimizer,
    #     data_loader=train_loader,
    #     # target_epsilon=8.0,
    #     # target_delta=1e-5,
    #     epochs=epochs,
    #     # max_grad_norm=1.0,
    # )
    vae_model, optimizer, train_loader = privacy_engine.make_private(
        module=vae_model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=2.0,
        max_grad_norm=1.0,
    )

    for epoch in range(epochs):

        avg_loss, avg_KL_s, avg_KL_z = 0.0, 0.0, 0.0
        avg_loss_val, avg_KL_s_val, avg_KL_z_val = 0.0, 0.0, 0.0
        samples_list, p_params_list, q_params_list, log_p_x_total, log_p_x_missing_total = [], [], [], [], []
        tau = max(1.0 - 0.01 * epoch, 1e-3)

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
                                                                                    miss_mask[:n_train_samples])
        
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
            if epoch % 100 == 0:
                visualization.print_loss(epoch, start_time, -avg_loss, avg_KL_s, avg_KL_z)
        

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
                print(f"Early stopping at epoch {epoch}.")
                break
        else:
            loss_val.append(torch.nan)

    if verbose:
        print("Training finished.")
    
    return vae_model, loss_train, loss_val


def train_HIVAE_bis(vae_model, data, miss_mask, true_miss_mask, feat_types_dict, batch_size, lr, epochs, verbose = True):

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

    # Training
    optimizer = optim.Adam(vae_model.parameters(), lr=lr)

    start_time = time.time()
    loss_train, error_observed_train, error_missing_train = [], [], []
    loss_val, error_observed_val, error_missing_val = [], [], []

    rng = np.random.default_rng(seed=42)
    # Setting for early stopping
    best_val_loss = float('inf')

    patience = 10 # 5
    n_iter_validation = 50
    n_iter_min = 100
    counter = 0

    dataset = MyCustomDataset(data_train, miss_mask_train, feat_types_dict)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) #, drop_last=True)

    for epoch in range(epochs):

        avg_loss, avg_KL_s, avg_KL_z = 0.0, 0.0, 0.0
        avg_loss_val, avg_KL_s_val, avg_KL_z_val = 0.0, 0.0, 0.0
        samples_list, p_params_list, q_params_list, log_p_x_total, log_p_x_missing_total = [], [], [], [], []
        tau = max(1.0 - 0.01 * epoch, 1e-3)

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
                                                                                    miss_mask[:n_train_samples])
        
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
            if epoch % 100 == 0:
                visualization.print_loss(epoch, start_time, -avg_loss, avg_KL_s, avg_KL_z)
        

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
                print(f"Early stopping at epoch {epoch}.")
                break
        else:
            loss_val.append(torch.nan)

    if verbose:
        print("Training finished.")
    
    return vae_model, loss_train, loss_val