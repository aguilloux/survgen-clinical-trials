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
from torchsurv.loss import weibull


def loglik_evaluation(batch_data_list, feat_types_list, miss_list, theta, normalization_params, n_generated_dataset=1, T_max=None):
    """
    Evaluates the log-likelihood of observed and missing data.

    Parameters:
    -----------
    batch_data_list : list of torch.Tensor
        List of tensors containing batch-wise feature data.
    
    feat_types_list : list of dict
        List of dictionaries specifying the type of each feature. Each dictionary should contain:
        - 'type': The type of the feature (e.g., 'real', 'cat', 'ordinal', etc.).
    
    miss_list : torch.Tensor
        A binary mask indicating observed (1) and missing (0) values in the dataset.
    
    theta : list of torch.Tensor
        List of parameter tensors associated with each feature for likelihood computation.
    
    normalization_params : list of dict
        List of normalization parameters for each feature, used in likelihood calculations.

    n_generated_dataset : int
        Number of samples to be generated per an input data point

    Returns:
    --------
    params_x : list of torch.Tensor
        List of estimated parameters for each feature.
    
    log_p_x : torch.Tensor
        Stacked log-likelihood values for observed data across all features.
    
    log_p_x_missing : torch.Tensor
        Stacked log-likelihood values for missing data across all features (used for test log-likelihood evaluation).
    
    samples_x : list of torch.Tensor
        List of sampled values for each feature from the estimated distributions.

    Notes:
    ------
    - The function dynamically calls the corresponding log-likelihood function from `loglik_models_missing_normalize`.
    - It supports various feature types by using `getattr()` to retrieve the appropriate function dynamically.
    """

    # Initialize storage lists
    params_x, log_p_x, log_p_x_missing, samples_x = [], [], [], []

    loglik_functions = {
        "real": loglik_real,
        "pos": loglik_pos,
        "count": loglik_count,
        "cat": loglik_cat,
        "ordinal": loglik_ordinal,
        "surv": loglik_surv,
        "surv_weibull": loglik_surv_weibull,
        "surv_loglog": loglik_surv_loglog,
        "surv_piecewise": loglik_surv_piecewise
    }

    # Compute log-likelihood for each feature type
    for feature_idx, batch_data in enumerate(batch_data_list):
        feature_type = feat_types_list[feature_idx]['type']
        
        # Dynamically select the appropriate likelihood function
        loglik_function = loglik_functions[feature_type]

        # Prepare input for likelihood function
        batch_data_ext = [batch_data, miss_list[:, feature_idx]]

        # Compute likelihood
        out = loglik_function(batch_data_ext, feat_types_list[feature_idx], theta[feature_idx], normalization_params[feature_idx], n_generated_dataset)

        # Store computed values
        params_x.append(out['params'])
        log_p_x.append(out['log_p_x'])
        log_p_x_missing.append(out['log_p_x_missing'])  # Test log-likelihood
        samples_x.append(out['samples'])

    # Stack log-likelihood tensors for efficient computation
    return params_x, torch.stack(log_p_x), torch.stack(log_p_x_missing), samples_x



def loglik_real(batch_data, list_type, theta, normalization_params, n_generated_dataset):
    """
    Computes the log-likelihood for real-valued (continuous) data under a Gaussian assumption.

    Parameters:
    -----------
    batch_data : tuple of (torch.Tensor, torch.Tensor)
        - `data`: Observed data tensor.
        - `missing_mask`: Binary mask tensor (1 = observed, 0 = missing).
    
    list_type : dict
        Dictionary specifying feature type and dimension.
    
    theta : tuple of (torch.Tensor, torch.Tensor)
        - `est_mean`: Predicted mean from the model.
        - `est_var`: Predicted variance from the model.
    
    normalization_params : tuple of (torch.Tensor, torch.Tensor)
        - `data_mean`: Mean used for normalizing the original dataset.
        - `data_var`: Variance used for normalizing the original dataset.

    n_generated_dataset : int
        Number of samples to be generated per an input data point

    Returns:
    --------
    output : dict
        Dictionary containing:
        - `params`: Estimated mean and variance.
        - `log_p_x`: Log-likelihood of observed data.
        - `log_p_x_missing`: Log-likelihood of missing data.
        - `samples`: Samples drawn from the estimated distribution.

    Notes:
    ------
    - The function ensures numerical stability using `torch.clamp()`.
    - A Gaussian distribution is assumed for the likelihood function.
    """

    # Extract data and missing mask
    data, missing_mask = batch_data
    missing_mask = missing_mask.float()
    epsilon = 1e-3  # Small constant for numerical stability

    # Retrieve normalization parameters and ensure stability
    data_mean, data_var = normalization_params
    # data_var = torch.clamp(data_var, min=epsilon)

    # Retrieve predicted mean and variance, ensuring variance is positive
    est_mean, est_var = theta
    est_var = torch.clamp(F.softplus(est_var), min=epsilon, max=1e20)

    # Perform affine transformation of the parameters
    est_mean = torch.sqrt(data_var) * est_mean + data_mean
    est_var = data_var * est_var

    # Compute log-likelihood using the Gaussian log-likelihood formula
    log_normalization = -0.5 * torch.log(torch.tensor(2 * torch.pi))
    log_variance_term = -0.5 * torch.sum(torch.log(est_var), dim=1)
    log_exponent = -0.5 * torch.sum((data - est_mean) ** 2 / est_var, dim=1)

    log_p_x = log_exponent + log_variance_term + log_normalization

    # Generate output dictionary
    output = {
        "params": [est_mean, est_var],
        "log_p_x": log_p_x * missing_mask,
        "log_p_x_missing": log_p_x * (1.0 - missing_mask),
        "samples": Normal(est_mean, torch.sqrt(est_var)).sample(sample_shape=(n_generated_dataset,))
    }

    return output


def loglik_surv(batch_data, list_type, theta, normalization_params, n_generated_dataset):
    """
    Computes the log-likelihood for survival data using a log-normal distribution.

    Parameters:
    -----------
    batch_data : tuple of (torch.Tensor, torch.Tensor)
        - `data`: Observed positive real-valued survival times and censoring indicators.
        - `missing_mask`: Binary mask (1 = observed, 0 = missing).

    list_type : dict
        Dictionary specifying feature types (not used in function but kept for compatibility).

    theta : tuple of (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor)
        - `est_mean_T`: Predicted log-mean for survival time.
        - `est_var_T`: Predicted log-variance for survival time.
        - `est_mean_C`: Predicted log-mean for censoring time.
        - `est_var_C`: Predicted log-variance for censoring time.

    normalization_params : tuple of (torch.Tensor, torch.Tensor)
        - `data_mean_log`: Log mean of the dataset.
        - `data_var_log`: Log variance of the dataset.

    n_generated_dataset : int
        Number of samples to generate per input data point.

    Returns:
    --------
    output : dict
        - `params`: Estimated mean and variance for survival and censoring.
        - `log_p_x`: Log-likelihood of observed data.
        - `log_p_x_missing`: Log-likelihood of missing data.
        - `samples`: Sampled values from the estimated log-normal distribution.
    """

    epsilon = 1e-5  # Small value for numerical stability

    # Extract normalization parameters
    data_mean_log, data_var_log = normalization_params

    # Extract observed data and mask
    data, missing_mask = batch_data
    T_surv, delta = data[:, 0].unsqueeze(1), data[:, 1].unsqueeze(1)
    data_log = torch.log1p(T_surv)  # Log-transform survival times
    missing_mask = missing_mask.float()

    # Extract predicted parameters and enforce positivity
    est_mean_T, est_var_T, est_mean_C, est_var_C = theta
    est_var_T = F.softplus(est_var_T).clamp(min=epsilon, max=1.0)
    est_var_C = F.softplus(est_var_C).clamp(min=epsilon, max=1.0)

    # Transform estimated parameters to original scale
    sqrt_var_log = torch.sqrt(data_var_log)
    est_mean_T = sqrt_var_log * est_mean_T + data_mean_log
    est_var_T *= data_var_log
    est_mean_C = sqrt_var_log * est_mean_C + data_mean_log
    est_var_C *= data_var_log

    
    def log_hazard_normal(t, mu, sigma):
        """
        Computes the hazard function h(t) for a normal distribution.

        Parameters:
        - t (Tensor): Time values at which to compute the hazard function.
        - mu (float or Tensor): Mean of the normal distribution.
        - sigma (float or Tensor): Standard deviation of the normal distribution.

        Returns:
        - h_t (Tensor): Hazard function values at time `t`.
        """
        # Ensure sigma is positive to avoid numerical issues
        sigma = torch.clamp(sigma, min=1e-5)

        # Normal distribution
        normal_dist = Normal(mu, sigma)

        # Compute PDF f(t)
        f_t = normal_dist.log_prob(t).exp()  # Equivalent to torch.exp(normal_dist.log_prob(t))

        # Compute survival function S(t) = 1 - CDF(t)
        S_t = 1 - normal_dist.cdf(t)

        # Compute hazard function h(t) = f(t) / S(t)
        h_t = f_t / torch.clamp(S_t, min=1e-10)  # Avoid division by zero

        log_h_t = torch.log(torch.clamp(h_t, min=1e-10))

        return log_h_t

    def cumulative_hazard_normal(t, mu, sigma):
        """
        Computes the cumulative hazard function H(t) for a normal distribution.

        Parameters:
        - t (Tensor): Time values at which to compute the cumulative hazard.
        - mu (float or Tensor): Mean of the normal distribution.
        - sigma (float or Tensor): Standard deviation of the normal distribution.

        Returns:
        - H_t (Tensor): Cumulative hazard values at time `t`.
        """
        # Ensure sigma is positive to avoid numerical issues
        sigma = torch.clamp(sigma, min=1e-5)

        # Normal distribution
        normal_dist = Normal(mu, sigma)

        # Compute the survival function S(t) = 1 - CDF
        S_t = 1 - normal_dist.cdf(t)

        # Compute cumulative hazard: H(t) = -log(S(t))
        H_t = -torch.log(torch.clamp(S_t, min=1e-10))  # Avoid log(0) issues

        return H_t

    log_p_x_T = delta * log_hazard_normal(data_log, est_mean_T, torch.sqrt(est_var_T)) - cumulative_hazard_normal(data_log, est_mean_T, torch.sqrt(est_var_T))
    log_p_x_C = (1 - delta) * log_hazard_normal(data_log, est_mean_C, torch.sqrt(est_var_C)) - cumulative_hazard_normal(data_log, est_mean_C, torch.sqrt(est_var_C))

    # Compute overall log-likelihood based on censoring indicator
    log_p_x = (log_p_x_T + log_p_x_C).sum(dim=1)

    # Generate samples from estimated log-normal distributions
    max_threshold = 2 * T_surv.max().item()
    def sample_from_log_normal(mean, var):
        return torch.clamp(
            torch.exp(Normal(mean, torch.sqrt(var)).sample((n_generated_dataset,))) - 1.0,
            min=0, max=max_threshold
        )

    sample_T = sample_from_log_normal(est_mean_T, est_var_T)
    sample_C = sample_from_log_normal(est_mean_C, est_var_C)

    return {
        "params": [est_mean_T, est_var_T, est_mean_C, est_var_C],
        "log_p_x": log_p_x * missing_mask,
        "log_p_x_missing": log_p_x * (1.0 - missing_mask),
        "samples": torch.cat((sample_T, sample_C), dim=-1),
    }


def loglik_surv_piecewise(batch_data, list_type, theta, normalization_params, n_generated_dataset):
    """
    Computes the log-likelihood for survival data

    Parameters:
    -----------
    batch_data : tuple of (torch.Tensor, torch.Tensor)
        - `data`: Observed positive real-valued survival times and censoring indicators.
        - `missing_mask`: Binary mask (1 = observed, 0 = missing).

    list_type : dict
        Dictionary specifying feature types (not used in function but kept for compatibility).

    theta : tuple of (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor)
        - `est_mean_T`: Predicted log-mean for survival time.
        - `est_var_T`: Predicted log-variance for survival time.
        - `est_mean_C`: Predicted log-mean for censoring time.
        - `est_var_C`: Predicted log-variance for censoring time.

    normalization_params : tuple of (torch.Tensor, torch.Tensor)
        - `data_mean_log`: Log mean of the dataset.
        - `data_var_log`: Log variance of the dataset.

    n_samples : int
        Number of samples to generate per input data point.

    Returns:
    --------
    output : dict
        - `params`: Estimated mean and variance for survival and censoring.
        - `log_p_x`: Log-likelihood of observed data.
        - `log_p_x_missing`: Log-likelihood of missing data.
        - `samples`: Sampled values from the estimated log-normal distribution.
    """

    # Extract observed data and mask
    data, missing_mask = batch_data
    missing_mask = missing_mask.float()

    # Extract normalization parameters
    data_min, data_max = normalization_params
    T_surv, delta = data[:, 0], data[:, 1]
    T_surv_scaled = (T_surv - data_min) / (data_max - data_min)

    # Extract predicted parameters and enforce positivity
    theta_T, theta_C, intervals = theta

    n, K = T_surv_scaled.shape[0], len(intervals)
    eps = 1e-8
    breaks = torch.cat((torch.Tensor([interval[0] for interval in intervals]), torch.Tensor([intervals[-1][1]])))
    bin_idx = torch.bucketize(T_surv_scaled, breaks[1:], right=False).flatten()  # shape: (n,)
    t0 = breaks[bin_idx]         # left edge of interval
    t1 = breaks[bin_idx + 1]     # right edge
    bin_width = (t1 - t0).clamp(min=eps)  # avoid division by zero

    density_T, density_C = F.softmax(theta_T, dim=1), F.softmax(theta_C, dim=1)
    cdf_T, cdf_C = torch.cumsum(density_T, dim=1), torch.cumsum(density_C, dim=1)
    surv_T, surv_C = 1.0 - cdf_T, 1.0 - cdf_C

    S0_T = torch.where(bin_idx == 0, torch.ones(n), surv_T[torch.arange(n), bin_idx - 1])
    S1_T = surv_T[torch.arange(n), bin_idx]
    S0_C = torch.where(bin_idx == 0, torch.ones(n), surv_C[torch.arange(n), bin_idx - 1])
    S1_C = surv_C[torch.arange(n), bin_idx]

    # Linear interpolation weight
    w = (T_surv_scaled - t0) / bin_width
    S_T_t = (1 - w) * S0_T + w * S1_T
    S_C_t = (1 - w) * S0_C + w * S1_C

    log_f_T = torch.log(density_T[torch.arange(n), bin_idx] / bin_width + eps)
    log_S_T = torch.log(S_T_t + eps)
    log_f_C = torch.log(density_C[torch.arange(n), bin_idx] / bin_width + eps)
    log_S_C = torch.log(S_C_t + eps)

    log_p_x_T = delta * log_f_T + (1 - delta) * log_S_T
    log_p_x_C = (1 - delta) * log_f_C + delta * log_S_C

    # Compute overall log-likelihood based on censoring indicator
    log_p_x = (log_p_x_T + log_p_x_C)

    # Generate samples from density function
    def sample_from_density(density, intervals, data_min, data_max):

        batch_size = density.shape[0]
        # Interval widths and start points
        intervals_tensor = torch.tensor(intervals, dtype=torch.float32)  # shape: (K, 2)
        starts = intervals_tensor[:, 0]  # shape: (K,)
        widths = intervals_tensor[:, 1] - intervals_tensor[:, 0]  # shape: (K,)

        # 1. Compute probability mass of each interval
        probs = density * widths
        probs = probs / probs.sum(dim=1, keepdim=True)  # normalize just in case

        # 2. Sample interval indices for each item
        interval_indices = torch.multinomial(probs, num_samples=1).squeeze(1)  # shape: (batch_size,)

        # 3. Sample uniformly within the selected intervals
        t_start = starts[interval_indices]        # (batch_size,)
        t_width = widths[interval_indices]        # (batch_size,)
        u = torch.rand(batch_size) # (batch_size,)

        samples = t_start + u * t_width           # sampled time in interval
        samples = data_min + (samples * (data_max - data_min)) / 1.0  # optional rescaling

        return samples.unsqueeze(1)  # shape: (batch_size, 1)

    sample_T, sample_C = [], []
    for _ in range(n_generated_dataset):
        T_s = sample_from_density(density_T, intervals, data_min, data_max)
        C_s = sample_from_density(density_C, intervals, data_min, data_max)
        mask = (T_s > data_max) & (C_s > data_max)
        C_s[mask] = data_max

        sample_T.append(T_s)
        sample_C.append(C_s)
    sample_T = torch.stack(sample_T, dim=0)
    sample_C = torch.stack(sample_C, dim=0)

    return {
        "params": [theta_T, theta_C],
        "log_p_x": log_p_x * missing_mask,
        "log_p_x_missing": log_p_x * (1.0 - missing_mask),
        "samples": torch.cat((sample_T, sample_C), dim=-1),
    }

def loglik_surv_weibull(batch_data, list_type, theta, normalization_params, n_generated_dataset):
    """
    Computes the log-likelihood for positive real-valued data using a Weibull distribution.

    Parameters:
    -----------
    batch_data : tuple of (torch.Tensor, torch.Tensor)
        - `data`: Observed positive real-valued data.
        - `missing_mask`: Binary mask (1 = observed, 0 = missing).

    list_type : dict
        Dictionary specifying feature type and dimension.

    theta : tuple of (torch.Tensor, torch.Tensor)
        - est_shape_T, est_scale_T, est_shape_C, est_scale_C

    normalization_params : tuple of (torch.Tensor, torch.Tensor)
        - data_min, data_max

    n_generated_dataset : int
        Number of samples to be generated per an input data point

    Returns:
    --------
    output : dict
        - `params`: Estimated mean and variance.
        - `log_p_x`: Log-likelihood of observed data.
        - `log_p_x_missing`: Log-likelihood of missing data.
        - `samples`: Sampled values from the estimated log-normal distribution.
    """
    min_shape = 1e-3
    min_scale = 1e-3
    max_shape = 1e3
    max_scale = 1e3


    # Extract data and mask
    data, missing_mask = batch_data
    missing_mask = missing_mask.float()

    # Extract normalization parameters
    data_min, data_max = normalization_params


    est_shape_T, est_scale_T, est_shape_C, est_scale_C = theta
    est_shape_T = F.softplus(est_shape_T).clamp(min=min_shape, max=max_shape)
    est_scale_T = F.softplus(est_scale_T).clamp(min=min_scale, max=max_scale)
    est_shape_C = F.softplus(est_shape_C).clamp(min=min_shape, max=max_shape)
    est_scale_C = F.softplus(est_scale_C).clamp(min=min_scale, max=max_scale)
    log_est_shape_T, log_est_scale_T, log_est_shape_C, log_est_scale_C = torch.log(est_shape_T), torch.log(est_scale_T), torch.log(est_shape_C), torch.log(est_scale_C)
    #log_est_scale_T = log_est_scale_T *  (data_max - data_min)
    #log_est_scale_C = log_est_scale_C *  (data_max - data_min)


    # Compute log-likelihood
    T_surv, delta = data[:, 0], data[:, 1]
    T_surv_scaled = (T_surv - data_min) / (data_max - data_min)
    log_p_x_T = delta * weibull.log_hazard(torch.stack([log_est_scale_T, log_est_shape_T]).T, T_surv_scaled, all_times=False) - weibull.cumulative_hazard(torch.stack([log_est_scale_T, log_est_shape_T]).T, T_surv_scaled, all_times=False)
    log_p_x_C = (1 - delta) * weibull.log_hazard(torch.stack([log_est_scale_C, log_est_shape_C]).T, T_surv_scaled, all_times=False) - weibull.cumulative_hazard(torch.stack([log_est_scale_C, log_est_shape_C]).T, T_surv_scaled, all_times=False)

    log_p_x = log_p_x_T + log_p_x_C

    sample_T, sample_C = [], []
    for _ in range(n_generated_dataset):
        U = torch.rand(T_surv.shape[0]).clamp(1e-6, 1)  # Avoid log(0)
        V = torch.rand(T_surv.shape[0]).clamp(1e-6, 1)  # Avoid log(0)
        T_sampled = est_scale_T * (-torch.log(U)) ** (1 / est_shape_T) * (data_max - data_min) / 1.0  + data_min
        C_sampled = est_scale_C * (-torch.log(V)) ** (1 / est_shape_C) * (data_max - data_min) / 1.0  + data_min
        sample_T.append(T_sampled)
        sample_C.append(C_sampled)

    max_threshold =  max(T_surv).item()
    # max_threshold = 1e20
    sample_T = torch.stack(sample_T, dim=0)#.clamp(0, max_threshold)
    sample_C = torch.stack(sample_C, dim=0).clamp(0, max_threshold)

    return {
        "params": [est_shape_T, est_scale_T, est_shape_C, est_scale_C],
        "log_p_x": log_p_x * missing_mask,
        "log_p_x_missing": log_p_x * (1.0 - missing_mask),
        "samples": torch.stack([sample_T, sample_C], dim=-1)
    }

def loglik_surv_loglog(batch_data, list_type, theta, normalization_params, n_generated_dataset):
    """
    Computes the log-likelihood for positive real-valued data using a log logistic distribution for the times and weibull distribution for the
    censoring times.

    Parameters:
    -----------
    batch_data : tuple of (torch.Tensor, torch.Tensor)
        - `data`: Observed positive real-valued data.
        - `missing_mask`: Binary mask (1 = observed, 0 = missing).

    list_type : dict
        Dictionary specifying feature type and dimension.

    theta : tuple of (torch.Tensor, torch.Tensor)
        - est_shape_T, est_scale_T, est_shape_C, est_scale_C

    normalization_params : tuple of (torch.Tensor, torch.Tensor)
        - data_min, data_max

    n_generated_dataset : int
        Number of samples to be generated per an input data point

    Returns:
    --------
    output : dict
        - `params`: Estimated mean and variance.
        - `log_p_x`: Log-likelihood of observed data.
        - `log_p_x_missing`: Log-likelihood of missing data.
        - `samples`: Sampled values from the estimated log-normal distribution.
    """
    min_shape = 1e-3
    min_scale = 1e-3
    max_shape = 1e3
    max_scale = 1e3


    # Extract data and mask
    data, missing_mask = batch_data
    missing_mask = missing_mask.float()

    # Extract normalization parameters
    data_min, data_max = normalization_params


    est_shapem1_T, est_scale_T, est_shape_C, est_scale_C = theta
    est_shapem1_T = F.softplus(est_shapem1_T).clamp(min=min_shape, max=max_shape)
    est_scale_T = F.softplus(est_scale_T).clamp(min=min_scale, max=max_scale)
    est_shape_C = F.softplus(est_shape_C).clamp(min=min_shape, max=max_shape)
    est_scale_C = F.softplus(est_scale_C).clamp(min=min_scale, max=max_scale)
    


    # get and scale the data
    T_surv, delta = data[:, 0], data[:, 1]
    T_surv_scaled = (T_surv - data_min) / (data_max - data_min)

    def log_hazard_loglog(t, scale, shapem1):
        """
        Computes the log hazard function h(t) for a loglogistic distribution.

        Parameters:
        - t (Tensor): Time values at which to compute the hazard function.
        - scale (float or Tensor): scale parameter aka alpha
        - shape (float or Tensor): shape parameter aka beta

        Returns:
        - log_h_t (Tensor): log Hazard function values at time `t`.
        """
        # Ensure parameters are positive to avoid numerical issues
        scale = torch.clamp(scale, min=1e-5)
        shape = shapem1 + 1#torch.clamp(shapem1, min=1e-5) + 1.

        # Compute hazard function h(t) 
        h_t = (shape/scale) * (t/scale) ** (shape-1)   / (1 + (t/scale) ** shape)  # Avoid division by zero

        log_h_t = torch.log(torch.clamp(h_t, min=1e-10))

        return log_h_t

    def cumulative_hazard_loglog(t, scale, shapem1):
        """
        Computes the cumulative hazard function H(t) for a loglogistic distribution.

        Parameters:
        - t (Tensor): Time values at which to compute the cumulative hazard.
        - scale (float or Tensor): scale parameter aka alpha
        - shape (float or Tensor): shape parameter aka beta

        Returns:
        - H_t (Tensor): Cumulative hazard values at time `t`.
        """
        # Ensure sigma is positive to avoid numerical issues
        scale = torch.clamp(scale, min=1e-5)
        shape = shapem1 + 1#torch.clamp(shapem1, min=1e-5) + 1.

        
        # Compute cumulative hazard: H(t) = -log(S(t))
        H_t = torch.log((1 + (t/scale) ** shape))  # Avoid log(0) issues

        return H_t

    def cdf_inv(u, scale, shapem1):
        """
        Computes the cumulative hazard function H(t) for a loglogistic distribution.

        Parameters:
        - t (Tensor): Time values at which to compute the cumulative hazard.
        - scale (float or Tensor): scale parameter aka alpha
        - shape (float or Tensor): shape parameter aka beta

        Returns:
        - H_t (Tensor): Cumulative hazard values at time `t`.
        """
        # Ensure sigma is positive to avoid numerical issues
        scale = torch.clamp(scale, min=1e-5)
        shape = shapem1 + 1#torch.clamp(shapem1, min=1e-5) + 1.


        # Compute cumulative hazard: H(t) = -log(S(t))
        Finv_u = scale * ((1-u)/u) ** (1 /shape) 

        return Finv_u

    # Compute log-likelihood
    log_est_shape_C, log_est_scale_C = torch.log(est_shape_C), torch.log(est_scale_C)
    

    # Compute log-likelihood

    log_p_x_T = delta * log_hazard_loglog(T_surv_scaled, est_scale_T, est_shapem1_T) - cumulative_hazard_loglog(T_surv_scaled, est_scale_T, est_shapem1_T)
    #log_p_x_C = (1 - delta) * weibull.log_hazard(torch.stack([log_est_scale_C, log_est_shape_C]).T, T_surv_scaled, all_times=False) - weibull.cumulative_hazard(torch.stack([log_est_scale_C, log_est_shape_C]).T, T_surv_scaled, all_times=False)
    log_p_x_C = (1 - delta) * log_hazard_loglog(T_surv_scaled, est_scale_C, est_shape_C) - cumulative_hazard_loglog(T_surv_scaled, est_scale_C, est_shape_C)

    log_p_x = log_p_x_T + log_p_x_C

    # generate
    sample_T, sample_C = [], []
    for _ in range(n_generated_dataset):
        
        U = torch.rand(T_surv.shape[0]).clamp(1e-6, 1)  
        V = torch.rand(T_surv.shape[0]).clamp(1e-6, 1)  
        
        T_sampled_scaled = cdf_inv(U,est_scale_T,est_shapem1_T)  
        C_sampled_scaled = est_scale_C * (-torch.log(V)) ** (1 / est_shape_C)
        T_sampled = T_sampled_scaled * (data_max - data_min)   + data_min 
        C_sampled = C_sampled_scaled * (data_max - data_min)   + data_min
        

        sample_T.append(T_sampled)
        sample_C.append(C_sampled)

    max_threshold =  data_max
    # max_threshold = 1e20
    sample_T = torch.stack(sample_T, dim=0)##.clamp(0, max_threshold)
    sample_C = torch.stack(sample_C, dim=0).clamp(0, max_threshold)

    return {
        "params": [est_shapem1_T, est_scale_T, est_shape_C, est_scale_C],
        "log_p_x": log_p_x * missing_mask,
        "log_p_x_missing": log_p_x * (1.0 - missing_mask),
        "samples": torch.stack([sample_T, sample_C], dim=-1)
    }    


def loglik_pos(batch_data, list_type, theta, normalization_params, n_generated_dataset):
    """
    Computes the log-likelihood for positive real-valued data using a log-normal distribution.

    Parameters:
    -----------
    batch_data : tuple of (torch.Tensor, torch.Tensor)
        - `data`: Observed positive real-valued data.
        - `missing_mask`: Binary mask (1 = observed, 0 = missing).
    
    list_type : dict
        Dictionary specifying feature type and dimension.
    
    theta : tuple of (torch.Tensor, torch.Tensor)
        - `est_mean`: Predicted log-mean.
        - `est_var`: Predicted log-variance.
    
    normalization_params : tuple of (torch.Tensor, torch.Tensor)
        - `data_mean_log`: Log mean of the dataset.
        - `data_var_log`: Log variance of the dataset.

    n_generated_dataset : int
        Number of samples to be generated per an input data point

    Returns:
    --------
    output : dict
        - `params`: Estimated mean and variance.
        - `log_p_x`: Log-likelihood of observed data.
        - `log_p_x_missing`: Log-likelihood of missing data.
        - `samples`: Sampled values from the estimated log-normal distribution.
    """
    epsilon = 1e-3

    # Extract normalization parameters
    data_mean_log, data_var_log = normalization_params
    # data_var_log = torch.clamp(data_var_log, min=epsilon)

    # Extract data and mask
    data, missing_mask = batch_data
    data_log = torch.log1p(data)  # Log transform
    missing_mask = missing_mask.float()

    # Extract predicted parameters and ensure positivity
    est_mean, est_var = theta
    est_var = F.softplus(est_var).clamp(min=epsilon, max=1.0)

    # Affine transformation of the parameters
    est_mean = torch.sqrt(data_var_log) * est_mean + data_mean_log
    est_var = data_var_log * est_var

    # Compute log-likelihood
    log_p_x = -0.5 * torch.sum((data_log - est_mean) ** 2 / est_var, dim=1) \
              - 0.5 * torch.sum(torch.log(2 * torch.pi * est_var), dim=1) \
              - torch.sum(data_log, dim=1)

    max_threshold = 2. * max(data).item()
    # max_threshold = 1e20
    return {
        "params": [est_mean, est_var],
        "log_p_x": log_p_x * missing_mask,
        "log_p_x_missing": log_p_x * (1.0 - missing_mask),
        "samples": torch.clamp(torch.exp(Normal(est_mean, torch.sqrt(est_var)).sample(sample_shape=(n_generated_dataset, ))) , min=0, max=max_threshold)
        #"samples": torch.clamp(torch.exp(Normal(est_mean, torch.sqrt(est_var)).sample(sample_shape=(n_generated_dataset, ))) - 1.0, min=0, max=max_threshold)
    }


def loglik_cat(batch_data, list_type, theta, normalization_params, n_generated_dataset):
    """
    Computes the log-likelihood for categorical data.

    Parameters:
    -----------
    batch_data : tuple of (torch.Tensor, torch.Tensor)
        - `data`: One-hot encoded categorical data.
        - `missing_mask`: Binary mask (1 = observed, 0 = missing).
    
    list_type : dict
        Dictionary specifying feature type and dimension.
    
    theta : torch.Tensor
        Logits for categorical distribution.

    n_generated_dataset : int
        Number of samples to be generated per an input data point

    Returns:
    --------
    output : dict
        - `params`: Logits for categorical distribution.
        - `log_p_x`: Log-likelihood of observed data.
        - `log_p_x_missing`: Log-likelihood of missing data.
        - `samples`: Sampled categorical values.
    """
    data, missing_mask = batch_data
    missing_mask = missing_mask.float()

    # Compute log-likelihood
    log_p_x = -F.cross_entropy(theta, data.argmax(dim=1), reduction='none')

    return {
        "params": theta,
        "log_p_x": log_p_x * missing_mask,
        "log_p_x_missing": log_p_x * (1.0 - missing_mask),
        "samples": F.one_hot(Categorical(logits=theta).sample(sample_shape=(n_generated_dataset, )), num_classes=int(list_type["nclass"]))
    }


def loglik_ordinal(batch_data, list_type, theta, normalization_params, n_generated_dataset):
    """
    Computes the log-likelihood for ordinal data using a cumulative distribution approach.

    Parameters:
    -----------
    batch_data : tuple of (torch.Tensor, torch.Tensor)
        - `data`: Ordinal data encoded as binary indicators.
        - `missing_mask`: Binary mask (1 = observed, 0 = missing).
    
    list_type : dict
        Dictionary specifying feature type and dimension.
    
    theta : tuple of (torch.Tensor, torch.Tensor)
        - `partition_param`: Parameters defining ordinal category partitions.
        - `mean_param`: Mean parameter for ordinal variable.

    n_generated_dataset : int
        Number of samples to be generated per an input data point

    Returns:
    --------
    output : dict
        - `params`: Estimated probabilities for each ordinal category.
        - `log_p_x`: Log-likelihood of observed data.
        - `log_p_x_missing`: Log-likelihood of missing data.
        - `samples`: Sampled ordinal values.
    """
    epsilon = 1e-6

    data, missing_mask = batch_data
    missing_mask = missing_mask.float()
    batch_size = data.shape[0]

    # Ensure increasing outputs for ordinal categories
    partition_param, mean_param = theta
    mean_value = mean_param.view(-1, 1)
    theta_values = torch.cumsum(torch.clamp(F.softplus(partition_param), min=epsilon, max=1e20), dim=1)

    sigmoid_est_mean = torch.sigmoid(theta_values - mean_value)
    mean_probs = torch.cat([sigmoid_est_mean, torch.ones((batch_size, 1))], dim=1) - \
                 torch.cat([torch.zeros((batch_size, 1)), sigmoid_est_mean], dim=1)

    mean_probs = torch.clamp(mean_probs, min=epsilon, max=1.0)

    # Compute log-likelihood
    true_values = F.one_hot(data.sum(dim=1).long() - 1, num_classes=int(list_type["nclass"]))
    log_p_x = -F.cross_entropy(torch.log(mean_probs), true_values.argmax(dim=1), reduction="none")

    # Generate samples from the ordinal distribution
    sampled_values = torch.distributions.Categorical(logits=mean_probs.log()).sample(sample_shape=(n_generated_dataset, ))
    samples = []
    for i in range(n_generated_dataset):
        samples.append((torch.arange(int(list_type["nclass"]), device=sampled_values.device)
                            .unsqueeze(0) < (sampled_values[i] + 1).unsqueeze(1)).float().unsqueeze(0))
    samples = torch.cat(samples, dim=0)
    
    return {
        "params": mean_probs,
        "log_p_x": log_p_x * missing_mask,
        "log_p_x_missing": log_p_x * (1.0 - missing_mask),
        "samples": samples
    }


def loglik_count(batch_data, list_type, theta, normalization_params, n_generated_dataset):
    """
    Computes the log-likelihood for count data using a Poisson distribution.

    Parameters:
    -----------
    batch_data : tuple of (torch.Tensor, torch.Tensor)
        - `data`: Observed count data.
        - `missing_mask`: Binary mask (1 = observed, 0 = missing).
    
    list_type : dict
        Dictionary specifying feature type and dimension.
    
    theta : torch.Tensor
        Predicted Poisson rate (lambda).

    n_generated_dataset : int
        Number of samples to be generated per an input data point

    Returns:
    --------
    output : dict
        - `params`: Poisson rate (lambda).
        - `log_p_x`: Log-likelihood of observed data.
        - `log_p_x_missing`: Log-likelihood of missing data.
        - `samples`: Sampled count values.
    """
    epsilon = 1e-6

    data, missing_mask = batch_data
    missing_mask = missing_mask.float()

    # Ensure lambda is positive
    est_lambda = torch.clamp(F.softplus(theta), min=epsilon, max=1e20)

    # Compute log-likelihood
    log_p_x = torch.sum(data * torch.log(est_lambda + 1e-8) - est_lambda - torch.lgamma(data + 1), dim=1)

    return {
        "params": est_lambda,
        "log_p_x": log_p_x * missing_mask,
        "log_p_x_missing": log_p_x * (1.0 - missing_mask),
        "samples": Poisson(est_lambda).sample(sample_shape=(n_generated_dataset, ))
    }