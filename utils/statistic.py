#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Converted to PyTorch

Created on Mon Feb 17 20:35:11 2025

@author: Van Tuan NGUYEN
"""

import torch
import torch.nn.functional as F
import numpy as np

def s_proposal_multinomial(X, s_layer, tau):
    """
    Proposes a categorical distribution for `s` using the Gumbel-Softmax trick.

    Parameters:
    -----------
    X : torch.Tensor
        Input feature tensor with shape `(batch_size, input_dim)`.
    
    self.s_layer : 
    
    tau : float
        Temperature parameter for Gumbel-Softmax.

    Returns:
    --------
    samples_s : torch.Tensor
        Sampled categorical latent variables using the Gumbel-Softmax trick.
    
    log_pi_aux : torch.Tensor
        Logits of the categorical distribution.
    """
    
    log_pi = s_layer(X)
    log_pi_aux = torch.log_softmax(log_pi, dim=-1)

    gumbel_noise = -torch.log(-torch.log(torch.rand_like(log_pi_aux)))
    samples_s = F.softmax((log_pi_aux + gumbel_noise) / tau, dim=-1)

    return samples_s, log_pi_aux

def z_prior_GMM(samples_s, z_distribution_layer):
    """
    Computes the Gaussian Mixture Model (GMM) prior for `z`.

    Parameters:
    -----------
    samples_s : torch.Tensor
        Sampled categorical latent variables `s`.
    
    z_distribution_layer : 

    Returns:
    --------
    mean_pz : torch.Tensor
        Mean of the Gaussian Mixture Model.
    
    log_var_pz : torch.Tensor
        Log variance (fixed to zero for standard normal prior).
    """
    
    mean_pz = z_distribution_layer(samples_s)
    log_var_pz = torch.zeros_like(mean_pz).clamp(min=-15.0, max=15.0)

    return mean_pz, log_var_pz


def z_proposal_GMM(X, samples_s, batch_size, z_dim, z_layer):
    """
    Proposes a Gaussian Mixture Model (GMM) for latent variable `z`.

    Parameters:
    -----------
    X : torch.Tensor
        Input feature tensor of shape `(batch_size, feature_dim)`.
    
    samples_s : torch.Tensor
        Sampled categorical latent variables of shape `(batch_size, s_dim)`.
    
    batch_size : int
        Number of samples in a batch.
    
    z_dim : int
        Dimensionality of the latent space `z`.

    z_layer :

    Returns:
    --------
    samples_z : torch.Tensor
        Sampled latent variables.
    
    list : [mean_qz, log_var_qz]
        - `mean_qz`: Mean of the latent `z` distribution.
        - `log_var_qz`: Log variance of the latent `z` distribution.
    """

    # Concatenate inputs
    concat_input = torch.cat([X, samples_s], dim=1)

    # Compute mean and log variance
    mean_qz, log_var_qz = torch.chunk(z_layer(concat_input), 2, dim=1) 

    # Avoid numerical instability
    log_var_qz = torch.clamp(log_var_qz, -15.0, 15.0)

    # Reparameterization trick
    eps = torch.randn((batch_size, z_dim), device=X.device)
    samples_z = mean_qz + torch.exp(0.5 * log_var_qz) * eps

    return samples_z, [mean_qz, log_var_qz]


def samples_concatenation(samples):
    """
    Concatenates multiple sample batches into a single dataset.

    Parameters:
    -----------
    samples : list of dict
        A list where each element is a dictionary containing batch-wise data with keys:
        - 'x': Feature data
        - 'y': Labels or target values
        - 'z': Latent variables
        - 's': Any additional variables

    Returns:
    --------
    samples_s : torch.Tensor
        Concatenated additional variables across batches.
    samples_z : torch.Tensor
        Concatenated latent variables across batches.
    samples_y : torch.Tensor
        Concatenated labels across batches.
    samples_x : torch.Tensor
        Concatenated feature data across batches.
    """
    
    samples_x = torch.cat([torch.cat(batch['x'], dim=-1) for batch in samples], dim=-2)
    samples_y = torch.cat([batch['y'] for batch in samples], dim=0)
    samples_z = torch.cat([batch['z'] for batch in samples], dim=0)
    samples_s = torch.cat([batch['s'] for batch in samples], dim=0)
    
    return samples_s, samples_z, samples_y, samples_x


def mean_imputation(train_data, miss_mask, types_dict):
    """
    Performs mean and mode imputation for missing values in categorical, ordinal, and continuous data.

    Parameters:
    -----------
    train_data : torch.Tensor
        The dataset containing missing values.
    
    miss_mask : torch.Tensor
        A binary mask indicating observed (1) and missing (0) values.
    
    types_dict : list of dict
        A list of dictionaries specifying the type and dimension of each feature.

    Returns:
    --------
    torch.Tensor
        The dataset with missing values imputed.
    """
    
    ind_ini, est_data = 0, []
    n_features = len(types_dict)

    for d in range(n_features):
        type_dict = types_dict[d]
        ind_end = ind_ini + (1 if type_dict['type'] in {'cat', 'ordinal'} else int(type_dict['dim']))
        miss_pattern = miss_mask[:, d] == 1  # Extract mask
        
        if type_dict['type'] in {'cat', 'ordinal'}:
            # Mode imputation
            values, counts = torch.unique(train_data[miss_pattern, ind_ini:ind_end], return_counts=True)
            data_mode = values[torch.argmax(counts)]  # Get the mode
        else:
            # Mean imputation
            data_mode = torch.mean(train_data[miss_pattern, ind_ini:ind_end], dim=0)
        
        # Apply imputation
        data_imputed = train_data[:, ind_ini:ind_end] * miss_mask[:, ind_ini:ind_end] + data_mode * (1 - miss_mask[:, ind_ini:ind_end])
        est_data.append(data_imputed)
        
        ind_ini = ind_end
    
    return torch.cat(est_data, dim=1)


def p_distribution_params_concatenation(params, types_dict):
    """
    Concatenates probability distribution parameters from multiple batches for p-distribution.

    Parameters:
    -----------
    params : list of dict
        A list of dictionaries where each dictionary contains distribution parameters for a batch.
    
    types_dict : list of dict
        A list of dictionaries specifying the type and dimension of each feature.

    Returns:
    --------
    out_dict : dict
        A dictionary containing concatenated probability distribution parameters across all batches.
    """
    
    keys = params[0].keys()
    out_dict = {key: params[0][key] for key in keys}  # Initialize with first batch

    for batch in params[1:]:  # Start from the second batch
        for key in keys:
            if key == 'y':
                out_dict[key] = torch.cat([out_dict[key], batch[key]], dim=0)
            
            elif key == 'z':
                out_dict[key] = (
                    torch.cat([out_dict[key][0], batch[key][0]], dim=0),
                    torch.cat([out_dict[key][1], batch[key][1]], dim=0)
                )

            elif key == 'x':
                for v, attr in enumerate(types_dict):
                    if attr['type'] in {'pos', 'real'}:
                        out_dict[key][v] = (
                            torch.cat([out_dict[key][v][0], batch[key][v][0]], dim=0),
                            torch.cat([out_dict[key][v][1], batch[key][v][1]], dim=0)
                        )
                    else:
                        out_dict[key][v] = torch.cat([out_dict[key][v], batch[key][v]], dim=0)
    
    return out_dict

def q_distribution_params_concatenation(params):
    """
    Concatenates probability distribution parameters from multiple batches for q-distribution.

    Parameters:
    -----------
    params : list of dict
        A list of dictionaries where each dictionary contains distribution parameters for a batch.

    Returns:
    --------
    out_dict : dict
        A dictionary containing concatenated probability distribution parameters across all batches.
    """

    keys = params[0].keys()
    out_dict = {key: params[0][key] if key == 'z' else [params[0][key]] for key in keys}

    for batch in params[1:]:  # Start from the second batch
        for key in keys:
            if key == 'z':
                out_dict[key] = (
                    torch.cat([out_dict[key][0], batch[key][0]], dim=0),
                    torch.cat([out_dict[key][1], batch[key][1]], dim=0)
                )
            else:
                out_dict[key].append(batch[key])

    # Concatenate 's' if it exists in the dictionary
    if 's' in out_dict:
        out_dict['s'] = torch.cat(out_dict['s'], dim=0)

    return out_dict


def mean_mode_like(loglik_params, types_dict):
    """
    Computes the mean and mode of various probability distributions based on log-likelihood parameters.

    Parameters:
    -----------
    loglik_params : list of torch.Tensors
        A list containing the log-likelihood parameters for each feature.
    
    types_dict : list of dict
        A list of dictionaries, each specifying the type and dimension of a feature.
        The dictionary should contain a key 'type' which can be:
        - 'real': Continuous real-valued data (assumed normally distributed).
        - 'pos': Positive continuous data (assumed log-normal distributed).
        - 'count': Discrete count data (assumed Poisson distributed).
        - 'cat' or 'ordinal': Categorical or ordinal data.

    Returns:
    --------
    loglik_mean : torch.Tensor
        The mean estimates for each feature based on its respective distribution.
    
    loglik_mode : torch.Tensor
        The mode estimates for each feature based on its respective distribution.
    """

    loglik_mean, loglik_mode = [], []

    for d, attrib in enumerate(loglik_params):
        feature_type = types_dict[d]['type']

        if feature_type == 'real':
            # Normal distribution: mean and mode are the same
            mean, mode = attrib[0], attrib[0]

        elif feature_type == 'pos':
            # Log-normal distribution
            exp_term = torch.exp(attrib[0])
            mean = torch.maximum(exp_term * torch.exp(0.5 * attrib[1]) - 1.0, torch.zeros(1))
            mode = torch.maximum(exp_term * torch.exp(-attrib[1]) - 1.0, torch.zeros(1))

        elif feature_type == 'count':
            # Poisson distribution: mean = lambda, mode = floor(lambda)
            mean, mode = attrib, torch.floor(attrib)

        else:
            # Categorical & ordinal: Mode imputation using argmax
            reshaped_mode = torch.reshape(torch.argmax(attrib, dim=1), (-1, 1))
            mean, mode = reshaped_mode, reshaped_mode
        
        loglik_mean.append(mean)
        loglik_mode.append(mode)

    loglik_mean = torch.squeeze(torch.cat(loglik_mean, dim=1))
    loglik_mode = torch.squeeze(torch.cat(loglik_mode, dim=1))

    return loglik_mean, loglik_mode


def error_computation(x_train, x_hat, types_dict, miss_mask):
    """
    Computes different error metrics (classification error, shift error, and RMSE)
    for observed and missing values based on feature types.

    Parameters:
    -----------
    x_train : torch.Tensor
        The ground truth data (actual values from the dataset).
    
    x_hat : torch.Tensor
        The predicted or imputed values.
    
    types_dict : list of dict
        A list of dictionaries where each dictionary describes a feature's type and dimension.
        The dictionary should contain a key 'type' which can be:
        - 'cat': Categorical data.
        - 'ordinal': Ordinal data.
        - Any other type is treated as continuous (real-valued).
    
    miss_mask : torch.Tensor
        A binary mask indicating missing values (1 = observed, 0 = missing).

    Returns:
    --------
    error_observed : list of torch.Tensors
        A list containing the errors computed on observed values for each feature.
    
    error_missing : list of torch.Tensors
        A list containing the errors computed on missing values for each feature.
    """

    error_observed = []
    error_missing = []
    ind_ini = 0

    for d, feature in enumerate(types_dict):
        feature_type = feature['type']
        ind_end = ind_ini + int(feature['dim'])

        # Masked values
        observed_mask = miss_mask[:, d] == 1
        missing_mask = miss_mask[:, d] == 0

        x_train_observed = x_train[observed_mask, ind_ini:ind_end]
        x_hat_observed = x_hat[observed_mask, ind_ini:ind_end]
        x_train_missing = x_train[missing_mask, ind_ini:ind_end]
        x_hat_missing = x_hat[missing_mask, ind_ini:ind_end]

        # Classification error (Categorical)
        if feature_type == 'cat':
            error_observed.append(torch.mean((x_train_observed != x_hat_observed).to(torch.float32)))
            error_missing.append(torch.mean((x_train_missing != x_hat_missing).to(torch.float32)) if torch.any(missing_mask) else 0)

        # Shift error (Ordinal)
        elif feature_type == 'ordinal':
            error_observed.append(torch.mean(torch.abs(x_train_observed - x_hat_observed)) / int(feature['nclass']))
            error_missing.append(torch.mean(torch.abs(x_train_missing - x_hat_missing)) / int(feature['nclass']) if torch.any(missing_mask) else 0)

        # Normalized RMSE (Continuous)
        elif feature_type == 'surv':
            x_hat_observed_ = x_hat_observed[:, 0] * x_train_observed[:, 0] + x_hat_observed[:, 1] * (1 - x_train_observed[:, 0])
            norm_term = torch.max(x_train_observed[:, 0]) - torch.min(x_train_observed[:, 0])
            error_observed.append(torch.sqrt(F.mse_loss(x_train_observed[:, 0], x_hat_observed_)) / norm_term)
            error_missing.append(torch.sqrt(F.mse_loss(x_train_missing, x_hat_missing)) / norm_term if torch.any(missing_mask) else 0)

        # Normalized RMSE (Continuous)
        else:
            norm_term = torch.max(x_train[:, d]) - torch.min(x_train[:, d])
            error_observed.append(torch.sqrt(F.mse_loss(x_train_observed, x_hat_observed)) / norm_term)
            error_missing.append(torch.sqrt(F.mse_loss(x_train_missing, x_hat_missing)) / norm_term if torch.any(missing_mask) else 0)

        ind_ini = ind_end  # Move to next feature index

    return torch.Tensor(error_observed), torch.Tensor(error_missing)