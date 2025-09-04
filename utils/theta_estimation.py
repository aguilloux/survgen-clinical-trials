#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Converted to PyTorch

Created on Mon Feb 17 20:35:11 2025

@author: Van Tuan NGUYEN
"""

import torch
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, Poisson


def theta_estimation_from_ys(samples_y, samples_s, feat_types_list, miss_list, theta_layer):
    """
    Estimates parameters (theta) for each feature type from `samples_y` and `samples_s`.

    Parameters:
    -----------
    samples_y : list of torch.Tensor
        List of partitioned `y` samples, where each entry corresponds to a feature.
    
    samples_s : torch.Tensor
        The latent state variable `s` tensor.
    
    feat_types_list : list of dict
        List specifying feature types and dimensions.
    
    miss_list : torch.Tensor
        A binary mask indicating observed (1) and missing (0) values.

    theta_layer :


    Returns:
    --------
    list :
        A list of estimated parameters (θ) for each feature.
    """
    
    # Mapping feature types to corresponding theta functions
    theta_functions = {
        "real": theta_real,
        "pos": theta_pos,
        "count": theta_count,
        "cat": theta_cat,
        "ordinal": theta_ordinal,
        "surv": theta_surv,
        "surv_weibull": theta_surv_weibull,
        "surv_loglog": theta_surv_loglog,
        "surv_piecewise": theta_surv_piecewise,
    }

    theta = []

    # Compute θ(x_d | y_d) for each feature type
    for i, y_sample in enumerate(samples_y):
        feature_type = feat_types_list[i]['type']

        # Partition the data into observed and missing based on mask
        mask = miss_list[:, i].bool()
        observed_y, missing_y = y_sample[mask], y_sample[~mask]
        observed_s, missing_s = samples_s[mask], samples_s[~mask]
        condition_indices = [~mask, mask]

        # Compute the corresponding theta function
        params = theta_functions[feature_type](observed_y, missing_y, observed_s, missing_s, condition_indices, theta_layer["feat_" + str(i)])
        theta.append(params)

    return theta



def theta_real(observed_y, missing_y, observed_s, missing_s, condition_indices, theta_layer):
    """
    Computes the mean and variance layers for real-valued data.

    This function estimates parameters for continuous real-valued features in a survival analysis model.

    Parameters:
    -----------
    observed_y : torch.Tensor
        Tensor of observed `y` values with shape `(batch_size, feature_dim)`.
    
    missing_y : torch.Tensor
        Tensor of missing `y` values with shape `(batch_size, feature_dim)`.
    
    observed_s : torch.Tensor
        Tensor of observed latent states `s` with shape `(batch_size, latent_dim)`.
    
    missing_s : torch.Tensor
        Tensor of missing latent states `s` with shape `(batch_size, latent_dim)`.
    
    condition_indices : list of lists
        Indices for observed and missing data.

    theta_layer : 

    Returns:
    --------
    list :
        `[h2_mean, h2_sigma]` where:
        - `h2_mean` is the estimated mean layer.
        - `h2_sigma` is the estimated variance layer.

    Notes:
    ------
    - This function uses `observed_data_layer` to apply a shared transformation to both observed and missing data.
    """

    # Mean layer
    h2_mean = observed_data_layer(
        torch.cat([observed_y, observed_s], dim=1),
        torch.cat([missing_y, missing_s], dim=1),
        condition_indices,
        layer=theta_layer["mean"]
    )

    # Sigma layer
    h2_sigma = observed_data_layer(
        observed_s,
        missing_s,
        condition_indices,
        layer=theta_layer["sigma"]
    )

    return [h2_mean, h2_sigma]


def theta_surv(observed_y, missing_y, observed_s, missing_s, condition_indices, theta_layer):
    """
    Computes the mean and standard deviation parameter of log-normal distribution for survival data (survival time and censoring time).

    This function estimates parameters for survival data in a survival analysis model.

    Parameters:
    -----------
    (Same as `theta_real`)

    Returns:
    --------
    list :
        `[h2_mean_T, h2_sigma_T, h2_mean_C, h2_sigma_C]` where:
        - `h2_mean_T` is the estimated mean layer for survival time.
        - `h2_sigma_T` is the estimated standard deviation layer for survival time.
        - `h2_mean_C` is the estimated mean layer for censoring time.
        - `h2_sigma_C` is the estimated standard deviation layer for censoring time.

    Notes:
    ------
    - Identical to `theta_real`, but tailored for **positive** real-valued survival data.
    """

    # Mean layer for survival time
    h2_mean_T = observed_data_layer(
        torch.cat([observed_y, observed_s], dim=1),
        torch.cat([missing_y, missing_s], dim=1),
        condition_indices,
        layer=theta_layer["mean_T"]
    )

    # Sigma layer for survival time
    h2_sigma_T = observed_data_layer(
        torch.cat([observed_s], dim=1),
        torch.cat([missing_s], dim=1),
        condition_indices,
        layer=theta_layer["sigma_T"]
    )

    # Mean layer for censoring time
    h2_mean_C = observed_data_layer(
        torch.cat([observed_y, observed_s], dim=1),
        torch.cat([missing_y, missing_s], dim=1),
        condition_indices,
        layer=theta_layer["mean_C"]
    )

    # Sigmalayer for censoring time
    h2_sigma_C = observed_data_layer(
        torch.cat([observed_s], dim=1),
        torch.cat([missing_s], dim=1),
        condition_indices,
        layer=theta_layer["sigma_C"]
    )

    return [h2_mean_T, h2_sigma_T, h2_mean_C, h2_sigma_C]

def theta_surv_weibull(observed_y, missing_y, observed_s, missing_s, condition_indices, theta_layer):
    """
    Computes the scale and shape parameters of Weibull distribution for survival data (survival time and censoring time).

    This function estimates parameters for survival data in a survival analysis model.

    Parameters:
    -----------
    (Same as `theta_real`)

    Returns:
    --------
    list :
        `[h2_shape_T, h2_scale_T, h2_shape_C, h2_scale_C]` where:
        - `h2_shape_T` is the estimated shape layer for survival time.
        - `h2_scale_T` is the estimated scale layer for survival time.
        - `h2_shape_C` is the estimated shape layer for censoring time.
        - `h2_scale_C` is the estimated scale layer for censoring time.

    Notes:
    ------
    - Identical to `theta_real`, but tailored for **positive** real-valued survival data.
    """
    h2_shape_T, h2_scale_T, h2_shape_C, h2_scale_C = observed_data_layer(
        torch.cat([observed_y, observed_s], dim=1),
        torch.cat([missing_y, missing_s], dim=1),
        condition_indices,
        layer=theta_layer["theta"]).T

    return [h2_shape_T, h2_scale_T, h2_shape_C, h2_scale_C]


def theta_surv_loglog(observed_y, missing_y, observed_s, missing_s, condition_indices, theta_layer):
    """
    Computes the scale and shape parameters of log logistic distribution for survival data (survival time and censoring time).

    This function estimates parameters for survival data in a survival analysis model.

    Parameters:
    -----------
    (Same as `theta_real`)

    Returns:
    --------
    list :
        `[h2_shape_T, h2_scale_T, h2_shape_C, h2_scale_C]` where:
        - `h2_shape_T` is the estimated shape layer for survival time.
        - `h2_scale_T` is the estimated scale layer for survival time.
        - `h2_shape_C` is the estimated shape layer for censoring time.
        - `h2_scale_C` is the estimated scale layer for censoring time.

    Notes:
    ------
    - Identical to `theta_real`, but tailored for **positive** real-valued survival data.
    """
    h2_shape_T, h2_scale_T, h2_shape_C, h2_scale_C = observed_data_layer(
        torch.cat([observed_y, observed_s], dim=1),
        torch.cat([missing_y, missing_s], dim=1),
        condition_indices,
        layer=theta_layer["theta"]).T

    return [h2_shape_T, h2_scale_T, h2_shape_C, h2_scale_C]

def theta_surv_piecewise(observed_y, missing_y, observed_s, missing_s, condition_indices, theta_layer):
    """
    Computes the density function for survival data (survival time and censoring time).

    This function estimates density function for survival data in a survival analysis model.

    Parameters:
    -----------
    (Same as `theta_real`)

    Returns:
    --------
    list :
        `[h2_shape_T, h2_scale_T, h2_shape_C, h2_scale_C]` where:
        - `h2_shape_T` is the estimated shape layer for survival time.
        - `h2_scale_T` is the estimated scale layer for survival time.
        - `h2_shape_C` is the estimated shape layer for censoring time.
        - `h2_scale_C` is the estimated scale layer for censoring time.

    Notes:
    ------
    - Identical to `theta_real`, but tailored for **positive** real-valued survival data.
    """
    h2_theta_T = observed_data_layer(
        torch.cat([observed_y, observed_s], dim=1),
        torch.cat([missing_y, missing_s], dim=1),
        condition_indices,
        layer=theta_layer["theta_T"])

    h2_theta_C = observed_data_layer(
        torch.cat([observed_y, observed_s], dim=1),
        torch.cat([missing_y, missing_s], dim=1),
        condition_indices,
        layer=theta_layer["theta_C"])

    intervals = theta_layer["intervals"]

    return [h2_theta_T, h2_theta_C, intervals]

def theta_pos(observed_y, missing_y, observed_s, missing_s, condition_indices, theta_layer):
    """
    Computes the mean and variance layers for positive real-valued data.

    This function estimates parameters for positive real-valued features in a survival analysis model.

    Parameters:
    -----------
    (Same as `theta_real`)

    Returns:
    --------
    list :
        `[h2_mean, h2_sigma]` where:
        - `h2_mean` is the estimated mean layer.
        - `h2_sigma` is the estimated variance layer.

    Notes:
    ------
    - Identical to `theta_real`, but tailored for **positive** real-valued data.
    """

    # Mean layer
    h2_mean = observed_data_layer(
        torch.cat([observed_y, observed_s], dim=1),
        torch.cat([missing_y, missing_s], dim=1),
        condition_indices,
        layer=theta_layer["mean"]
    )

    # Sigma layer
    h2_sigma = observed_data_layer(
        observed_s,
        missing_s,
        condition_indices,
        layer=theta_layer["sigma"]
    )

    return [h2_mean, h2_sigma]


def theta_count(observed_y, missing_y, observed_s, missing_s, condition_indices, theta_layer):
    """
    Computes the lambda layer for count-valued data.

    This function estimates the rate parameter (lambda) for Poisson-distributed count data.

    Parameters:
    -----------
    (Same as `theta_real`)

    Returns:
    --------
    torch.Tensor :
        The estimated lambda layer (`h2_lambda`).

    Notes:
    ------
    - Used for modeling **count-based** survival features.
    - Applies a **linear transformation** using `observed_data_layer` to compute `lambda`.
    """

    h2_lambda = observed_data_layer(
        torch.cat([observed_y, observed_s], dim=1),
        torch.cat([missing_y, missing_s], dim=1),
        condition_indices,
        layer=theta_layer
    )

    return h2_lambda


def theta_cat(observed_y, missing_y, observed_s, missing_s, condition_indices, theta_layer):
    """
    Computes the log-probability layer for categorical data.

    This function estimates log-probabilities for categorical features using a linear layer.

    Parameters:
    -----------
    (Same as `theta_real`)

    Returns:
    --------
    torch.Tensor :
        Log-probability tensor (`h2_log_pi`) with shape `(batch_size, num_classes)`.

    Notes:
    ------
    - Uses `observed_data_layer` to compute logits for **all but one** class.
    - Ensures **identifiability** by appending a **zero log-probability** for the first category.
    """
    
    h2_log_pi_partial = observed_data_layer(
        torch.cat([observed_y, observed_s], dim=1),
        torch.cat([missing_y, missing_s], dim=1),
        condition_indices,
        layer=theta_layer
    )

    # Ensure the first value is zero for identifiability
    h2_log_pi = torch.cat([torch.zeros((h2_log_pi_partial.shape[0], 1)), h2_log_pi_partial], dim=1)

    return h2_log_pi


def theta_ordinal(observed_y, missing_y, observed_s, missing_s, condition_indices, theta_layer):
    """
    Computes the partitioning and mean layers for ordinal data.

    This function estimates parameters for ordinal features using a cumulative distribution approach.

    Parameters:
    -----------
    (Same as `theta_real`)

    Returns:
    --------
    list :
        `[h2_theta, h2_mean]` where:
        - `h2_theta` represents the estimated partitioning layer.
        - `h2_mean` is the estimated mean layer.

    Notes:
    ------
    - `h2_theta` defines the **ordered category partitions**.
    - `h2_mean` estimates **underlying latent scores** for ordinal regression.
    """

    # Theta layer
    h2_theta = observed_data_layer(
        observed_s,
        missing_s,
        condition_indices,
        layer=theta_layer["theta"]
    )

    # Mean layer
    h2_mean = observed_data_layer(
        torch.cat([observed_y, observed_s], dim=1),
        torch.cat([missing_y, missing_s], dim=1),
        condition_indices,
        layer=theta_layer["mean"]
    )

    return [h2_theta, h2_mean]


def observed_data_layer(observed_data, missing_data, condition_indices, layer=None):
    """
    Train a layer with the observed data and reuse it for the missing data in PyTorch.

    Parameters:
    -----------
    observed_data : torch.Tensor
        A tensor containing the observed (non-missing) data.
    
    missing_data : torch.Tensor
        A tensor containing the missing data.
    
    condition_indices : list of lists
        A list containing two lists:
        - `condition_indices[0]`: Indices corresponding to missing data.
        - `condition_indices[1]`: Indices corresponding to observed data.

    layer : 

    Returns:
    --------
    torch.Tensor
        A tensor combining both observed and missing data outputs after transformation.

    """

    # Forward pass for observed data
    obs_output = layer(observed_data)

    # Forward pass for missing data (using same layer but without updates)
    with torch.no_grad():
        miss_output = layer(missing_data)

    # Combine outputs based on condition indices
    output = torch.empty_like(torch.cat([miss_output, obs_output], dim=0))
    output[condition_indices[0]] = miss_output  # Missing data indices
    output[condition_indices[1]] = obs_output   # Observed data indices

    return output