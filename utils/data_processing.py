#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Converted to PyTorch

Created on Mon Feb 17 20:35:11 2025

@author: Van Tuan NGUYEN
"""

import argparse
import os
import csv
import numpy as np
import torch
import pandas as pd


def get_args(argv = None):
    parser = argparse.ArgumentParser(description='Default parameters of the models',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', type=int, default=200, help='Size of the batches')
    parser.add_argument('--epochs',type=int,default=5001, help='Number of epochs of the simulations')
    parser.add_argument('--perp',type=int,default=10, help='Perplexity for the t-SNE')
    parser.add_argument('--train', type=int,default=1, help='Training model flag')
    parser.add_argument('--display', type=int,default=1, help='Display option flag')
    parser.add_argument('--save', type=int,default=1000, help='Save variables every save iterations')
    parser.add_argument('--restore', type=int,default=0, help='To restore session, to keep training or evaluation') 
    parser.add_argument('--plot', type=int,default=1, help='Plot results flag')
    parser.add_argument('--dim_latent_s',type=int,default=10, help='Dimension of the categorical space')
    parser.add_argument('--dim_latent_z',type=int,default=2, help='Dimension of the Z latent space')
    parser.add_argument('--dim_latent_y',type=int,default=10, help='Dimension of the Y latent space')
    parser.add_argument('--dim_latent_y_partition',type=int, nargs='+', help='Partition of the Y latent space')
    parser.add_argument('--miss_percentage_train',type=float,default=0.0, help='Percentage of missing data in training')
    parser.add_argument('--miss_percentage_test',type=float,default=0.0, help='Percentage of missing data in test')
    parser.add_argument('--model_name', type=str, default='model_new', help='File of the training model')
    parser.add_argument('--save_file', type=str, default='new_mnist_zdim5_ydim10_4images_', help='Save file name')
    parser.add_argument('--data_file', type=str, default='MNIST_data', help='File with the data')
    parser.add_argument('--types_file', type=str, default='mnist_train_types2.csv', help='File with the types of the data')
    parser.add_argument('--miss_file', type=str, default='Missing_test.csv', help='File with the missing indexes mask')
    parser.add_argument('--true_miss_file', type=str, help='File with the missing indexes when there are NaN in the data')
    
    return parser.parse_args(argv)

def read_data(data_file, types_file, miss_file, true_miss_file, surv_type=None, return_cat_mapping=False):
    """
    Reads data from CSV files, handles missing values, and applies necessary transformations.

    Parameters:
    -----------
    data_file : str
        Path to the CSV file containing the dataset.
    
    types_file : str
        Path to the CSV file specifying the data types and dimensions for each feature.
    
    miss_file : str
        Path to the CSV file indicating the missing values in the dataset.
    
    true_miss_file : str or None
        Path to the CSV file containing the true missing value mask, if available.

    surv_type : str, default=None
        Type identifier for the survival outcome.

    return_cat_mapping : bool, default=False
        If True, additionally return a dict mapping each 'cat' feature name to its
        {original_value: integer_code} mapping.

    Returns:
    --------
    data : torch.Tensor
        Transformed dataset with categorical, ordinal, and continuous values properly encoded.
    
    types_dict : list of dict
        A list of dictionaries specifying the type and dimension of each feature.
    
    miss_mask : torch.Tensor
        A binary mask indicating observed (1) and missing (0) values.
    
    true_miss_mask : torch.Tensor
        A binary mask indicating the actual missing values, if provided.
    
    n_samples : int
        The number of samples in the dataset.

    cat_mapping : dict
        Only returned when return_cat_mapping=True. Maps each 'cat' feature name to a
        {original_value: integer_code} dict, with codes assigned in sorted/alphabetical
        order (the same ordering the one-hot encoder derives via torch.unique).
    """
    
    # Read types of data from types file
    with open(types_file) as f:
        types_dict = [{k: v for k, v in row.items()} for row in csv.DictReader(f, skipinitialspace=True)]
    if surv_type is not None:
        for i in range(len(types_dict)):
            if types_dict[i]["name"] == "survcens":
                types_dict[i]["type"] = surv_type

    if return_cat_mapping:
        # Read data from input file. Categorical ('cat') columns may hold text labels, which
        # are mapped to integer codes 0..k-1 in sorted/alphabetical order -- the same ordering
        # the one-hot encoder below derives via torch.unique. NaNs are preserved so the
        # missing-value handling further down is unaffected.
        raw = pd.read_csv(data_file, header=None)

        cat_mapping = {}
        col = 0
        for feature in types_dict:
            if feature['type'] == 'cat':
                codes, uniques = pd.factorize(raw[col], sort=True)
                codes = codes.astype(np.float32)
                codes[codes == -1] = np.nan  # restore missing values dropped by factorize
                raw[col] = codes
                cat_mapping[feature['name']] = {
                    (v.item() if hasattr(v, 'item') else v): i for i, v in enumerate(uniques)
                }
                col += 1
            elif feature['type'] in ['surv', 'surv_weibull', 'surv_loglog', 'surv_piecewise']:
                col += 2  # survival outcome occupies two columns
            else:
                col += 1

        data = torch.tensor(raw.to_numpy(dtype=np.float32), dtype=torch.float32)
    else:
        with open(data_file, 'r') as f:
            data = [[float(x) for x in rec] for rec in csv.reader(f, delimiter=',')]
            data = torch.tensor(data, dtype=torch.float32)
    
    # Handle true missing values if provided
    if true_miss_file:
        with open(true_miss_file, 'r') as f:
            missing_positions = [[int(x) for x in rec] for rec in csv.reader(f, delimiter=',')]
            missing_positions = torch.tensor(missing_positions, dtype=torch.long)

        true_miss_mask = torch.ones((data.shape[0], len(types_dict)))
        true_miss_mask[missing_positions[:, 0] - 1, missing_positions[:, 1] - 1] = 0  # CSV indexes start at 1
        
        # Replace NaNs with appropriate default values
        nan_mask = torch.isnan(data)
        data_filler = torch.zeros(data.shape[1], dtype=torch.float32)
        
        for i, dtype in enumerate(types_dict):
            if dtype['type'] in {'cat', 'ordinal'}:
                unique_vals = torch.unique(data[:, i][~nan_mask[:, i]])  # Get unique non-NaN values
                data_filler[i] = unique_vals[0] if len(unique_vals) > 0 else 0  # Fill with first category
            else:
                data_filler[i] = 0.0  # Fill numerical data with 0
        
        data[nan_mask] = data_filler.repeat(data.shape[0], 1)[nan_mask]
    
    else:
        true_miss_mask = torch.ones((data.shape[0], len(types_dict)))  # No effect on data if no file is provided
    
    # Construct processed data matrices
    data_complete = []
    
    feat_idx = 0
    feat_names = []
    for i, feature in enumerate(types_dict):

        if feature['type'] == 'cat':
            # One-hot encoding for categorical data
            cat_data = data[:, feat_idx].to(torch.int64)
            unique_vals, indexes = torch.unique(cat_data, return_inverse=True)
            new_categories = torch.arange(int(feature['nclass']), dtype=torch.int64)
            mapped_categories = new_categories[indexes]
            
            one_hot = torch.zeros((data.shape[0], len(new_categories)))
            one_hot[torch.arange(data.shape[0]), mapped_categories] = 1
            data_complete.append(one_hot)
            feat_names += [feature['name'] + "_" + str(j) for j in np.arange(len(new_categories))]
        
        elif feature['type'] == 'ordinal':
            # Thermometer encoding for ordinal data
            ordinal_data = data[:, feat_idx].to(torch.int64)
            unique_vals, indexes = torch.unique(ordinal_data, return_inverse=True)
            new_categories = torch.arange(int(feature['nclass']), dtype=torch.int64)
            mapped_categories = new_categories[indexes]
            
            thermometer = torch.zeros((data.shape[0], len(new_categories) + 1))
            thermometer[:, 0] = 1
            thermometer[torch.arange(data.shape[0]), 1 + mapped_categories] = -1
            thermometer = torch.cumsum(thermometer, dim=1)

            data_complete.append(thermometer[:, :-1])  # Exclude last column
            feat_names += [feature['name'] + "_" + str(j) for j in np.arange(len(new_categories))]

        elif feature['type'] == 'count':
            # Shift zero-based counts if necessary
            count_data = data[:, feat_idx].unsqueeze(1)
            if torch.min(count_data) == 0:
                count_data += 1
            data_complete.append(count_data)
            feat_names += [feature['name']]

        elif feature['type'] in ['surv', 'surv_weibull', 'surv_loglog', 'surv_piecewise']:
            # Survival data take two columns
            data_complete.append(data[:, feat_idx : feat_idx + 2])
            feat_idx += 1
            feat_names += ["time", "censor"]
        
        else:
            # Keep continuous data as is
            data_complete.append(data[:, feat_idx].unsqueeze(1))
            feat_names += [feature['name']]
    
        feat_idx += 1
    # Concatenate processed features
    data = torch.cat(data_complete, dim=1)
    df = pd.DataFrame(data, columns=feat_names)

    # Read missing mask file
    n_samples, n_variables = data.shape[0], len(types_dict)
    miss_mask = torch.ones((n_samples, n_variables))

    if os.path.isfile(miss_file):
        with open(miss_file, 'r') as f:
            missing_positions = [[int(x) for x in rec] for rec in csv.reader(f, delimiter=',')]
            missing_positions = torch.tensor(missing_positions, dtype=torch.long)
        if missing_positions.numel() != 0:
            miss_mask[missing_positions[:, 0] - 1, missing_positions[:, 1] - 1] = 0  # CSV indexes start at 1
    
    if return_cat_mapping:
        return df, types_dict, miss_mask, true_miss_mask, n_samples, cat_mapping
    return df, types_dict, miss_mask, true_miss_mask, n_samples



def next_batch(data, types_dict, miss_mask, batch_size, index_batch):
    """
    Generates the next minibatch of data and splits it into its respective features.

    Parameters:
    -----------
    data : torch.Tensor
        The complete dataset from which to extract a batch.
    
    types_dict : list of dict
        A list of dictionaries specifying the type and dimension of each feature.
    
    miss_mask : torch.Tensor
        A binary mask indicating missing values (1 = observed, 0 = missing).
    
    batch_size : int
        The number of samples to include in each batch.
    
    index_batch : int
        The index of the current batch to extract.

    Returns:
    --------
    data_list : list of torch.Tensors
        A list containing feature-wise separated data for the current batch.
    
    miss_list : torch.Tensor
        The corresponding missing data mask for the current batch.
    """
    
    # Extract minibatch
    batch_xs = data[index_batch * batch_size : (index_batch + 1) * batch_size, :]
    
    # Split variables in the batch
    data_list, initial_index = [], 0
    for d in types_dict:
        dim = (int(d['nclass']) if d["type"] in ['cat', 'ordinal'] else int(d['dim']))
        data_list.append(batch_xs[:, initial_index : initial_index + dim])
        initial_index += dim
    
    # Extract missing mask for the batch
    miss_list = miss_mask[index_batch * batch_size : (index_batch + 1) * batch_size, :]
    
    return data_list, miss_list

def load_data_types(types_file):
    """
    Reads the types of data from a CSV file and returns a dictionary.

    Parameters:
    -----------
    types_file : str
        Path to the CSV file containing variable types.

    Returns:
    --------
    list of dict:
        A list where each dictionary specifies the type of a variable.
    """
    with open(types_file, newline='') as f:
        return [{k: v for k, v in row.items()} for row in csv.DictReader(f, skipinitialspace=True)]


def resolve_feat_normalization_globals(globals_list, feat_types_list, norm_mode):
    """
    Select which frozen statistics to keep for the requested normalization
    mode, given the full per-feature globals from
    compute_feat_normalization_globals.

    Parameters
    ----------
    globals_list : list
        Full per-feature statistics (one entry per feature, or None for
        types that need none), as returned by
        compute_feat_normalization_globals.
    feat_types_list : list of dict
        Feature descriptors, same order as globals_list. Kept for API
        symmetry with 'global' mode; not needed to build the 'batch' list.
    norm_mode : {'global', 'batch'}
        'global' : every feature keeps its frozen global statistics
                   (globals_list returned unchanged).
        'batch'  : every feature falls back to per-batch statistics — each
                   entry is set to None so normalize_features recomputes the
                   statistics (mean/var for the z-score families, min/max for
                   the survival families) from each batch. count / cat /
                   ordinal carry no statistics (already None).

    Returns
    -------
    list
        The per-feature statistics list to store on
        vae_model.feat_normalization_globals.

    Raises
    ------
    ValueError
        If norm_mode is not 'global' or 'batch'.
    """
    if norm_mode == "global":
        return globals_list
    if norm_mode == "batch":
        return [None] * len(globals_list)
    raise ValueError(f"norm_mode must be 'global' or 'batch', got {norm_mode!r}.")


def compute_feat_normalization_globals(data, feat_types_list, miss_mask):
    """
    Computes frozen normalization statistics on the full (training) set,
    one entry per feature, matching the order of feat_types_list. Call
    once before training (after combining miss_mask with true_miss_mask)
    and assign the result to vae_model.feat_normalization_globals so the
    same scale is used by every forward pass during training, validation
    and generation.

    Returned list elements (each statistic is a 0-dim torch.Tensor so
    downstream likelihood code can call torch.sqrt etc. on it directly):
      - 'real'             : (mean, var)          on observed values
      - 'pos'              : (log_mean, log_var)  on log1p(observed values)
      - 'surv'             : (log_mean, log_var)  on log1p(observed time column)
      - 'surv_weibull',
        'surv_loglog',
        'surv_piecewise'   : (data_min, data_max) on observed time column,
                             with data_min = observed_min - 1e-3
      - 'count', 'cat',
        'ordinal'          : None (no statistics needed)

    Parameters
    ----------
    data : torch.Tensor
        Full data tensor of shape (N, sum(feature_dims)). Typically the
        training-set slice.
    feat_types_list : list of dict
        Feature descriptors as used throughout the model.
    miss_mask : torch.Tensor
        Mask of shape (N, n_features) with 1 = observed, 0 = missing.
        Pass the already-combined (miss_mask × true_miss_mask) tensor.
    """
    n = data.shape[0]
    data_list, miss_list = next_batch(data, feat_types_list, miss_mask, n, 0)

    globals_list = []
    for i, feat in enumerate(feat_types_list):
        observed_mask = miss_list[:, i] == 1
        d = data_list[i]
        feature_type = feat['type']

        if feature_type == 'real':
            obs = d[observed_mask]
            data_var, data_mean = torch.var_mean(obs, unbiased=False)
            data_var = torch.clamp(data_var, min=1e-6, max=1e20)
            globals_list.append((data_mean.detach(), data_var.detach()))

        elif feature_type == 'pos':
            obs_log = torch.log1p(d[observed_mask])
            data_var_log, data_mean_log = torch.var_mean(obs_log, unbiased=False)
            data_var_log = torch.clamp(data_var_log, min=1e-6, max=1e20)
            globals_list.append((data_mean_log.detach(), data_var_log.detach()))

        elif feature_type == 'surv':
            time_obs_log = torch.log1p(d[observed_mask, 0])
            data_var_log, data_mean_log = torch.var_mean(time_obs_log, unbiased=False)
            data_var_log = torch.clamp(data_var_log, min=1e-6, max=1e20)
            globals_list.append((data_mean_log.detach(), data_var_log.detach()))

        elif feature_type in ('surv_weibull', 'surv_loglog', 'surv_piecewise'):
            time_obs = d[observed_mask, 0]
            data_min = (time_obs.min() - 1e-3).detach()
            data_max = time_obs.max().detach()
            globals_list.append((data_min, data_max))

        else:
            # count, cat, ordinal — no statistics needed
            globals_list.append(None)

    return globals_list


def normalize_features(batch_data_list, feat_types_list, miss_list, feat_normalization_globals=None):
    """
    Normalizes input features for the encoder, using frozen training-set
    statistics when `feat_normalization_globals` is provided.

    This is *not* nn.BatchNorm. It performs type-aware feature
    standardization on input data (z-score for real / pos / legacy surv,
    min-max scaling for the parametric survival families, log-transform
    for count, pass-through for cat / ordinal). When the frozen-globals
    argument is provided, statistics are NOT recomputed from the current
    batch — this both prevents train/inference scale drift and avoids the
    per-sample-gradient coupling that would otherwise invalidate any
    differentially-private training.

    Parameters
    ----------
    batch_data_list : list of torch.Tensor
        List of input data tensors, each corresponding to a feature.
    feat_types_list : list of dict
        List specifying the type of each feature.
    miss_list : torch.Tensor
        Binary mask indicating observed (1) and missing (0) values.
    feat_normalization_globals : list or None
        Per-feature frozen statistics produced by
        compute_feat_normalization_globals (one entry per feature, or
        None for types that need no stats). When this argument itself is
        None, the function falls back to per-batch statistics. The
        fallback exists only for pre-training / diagnostic forward passes
        — trained models (and especially DP-trained models) should always
        pass the frozen globals.

    Returns
    -------
    normalized_data : list of torch.Tensor
        List of normalized feature tensors.
    normalization_parameters : list of tuples
        Per-feature normalization parameters used by the decoder
        likelihood to un-normalize predictions.
    """

    normalized_data = []
    normalization_parameters = []

    for i, d in enumerate(batch_data_list):
        observed_mask = miss_list[:, i] == 1
        missing_mask = ~observed_mask
        observed_data = d[observed_mask]

        feature_type = feat_types_list[i]['type']
        feat_globals = (
            feat_normalization_globals[i]
            if feat_normalization_globals is not None
            else None
        )

        if feature_type == 'real':
            if feat_globals is not None:
                data_mean, data_var = feat_globals
            else:
                data_var, data_mean = torch.var_mean(observed_data, unbiased=False)
                data_var = torch.clamp(data_var, min=1e-6, max=1e20)

            normalized_observed = (observed_data - data_mean) / torch.sqrt(data_var)
            normalized_d = torch.zeros_like(d)
            normalized_d[observed_mask] = normalized_observed
            normalized_d[missing_mask] = 0

            normalization_parameters.append((data_mean, data_var))

        elif feature_type == 'pos':
            observed_data_log = torch.log1p(observed_data)
            if feat_globals is not None:
                data_mean_log, data_var_log = feat_globals
            else:
                data_var_log, data_mean_log = torch.var_mean(observed_data_log, unbiased=False)
                data_var_log = torch.clamp(data_var_log, min=1e-6, max=1e20)

            normalized_observed = (observed_data_log - data_mean_log) / torch.sqrt(data_var_log)
            normalized_d = torch.zeros_like(d)
            normalized_d[observed_mask] = normalized_observed
            normalized_d[missing_mask] = 0

            normalization_parameters.append((data_mean_log, data_var_log))

        elif feature_type == 'count':
            normalized_d = torch.zeros_like(d)
            normalized_d[observed_mask] = torch.log1p(observed_data)
            normalized_d[missing_mask] = 0

            normalization_parameters.append((0.0, 1.0))

        elif feature_type == 'surv':
            observed_data_log = torch.log1p(observed_data[:, 0])
            if feat_globals is not None:
                data_mean_log, data_var_log = feat_globals
            else:
                data_var_log, data_mean_log = torch.var_mean(observed_data_log, unbiased=False)
                data_var_log = torch.clamp(data_var_log, min=1e-6, max=1e20)

            normalized_observed = (observed_data_log - data_mean_log) / torch.sqrt(data_var_log)
            normalized_d = torch.zeros_like(d)
            normalized_d[observed_mask, 0] = normalized_observed
            normalized_d[observed_mask, 1] = observed_data[:, 1]
            normalized_d[missing_mask] = 0

            normalization_parameters.append((data_mean_log, data_var_log))

        elif feature_type == 'surv_weibull':
            if feat_globals is not None:
                data_min, data_max = feat_globals
            else:
                data_min = torch.min(observed_data[:, 0]) - 1e-3
                data_max = torch.max(observed_data[:, 0])

            normalization_parameters.append((data_min, data_max))

            normalized_observed = observed_data[:, 0] / data_max
            normalized_d = torch.zeros_like(d)
            normalized_d[observed_mask, 0] = normalized_observed
            normalized_d[observed_mask, 1] = observed_data[:, 1]
            normalized_d[missing_mask] = 0
        
        elif feature_type in ('surv_loglog', 'surv_piecewise'):
            if feat_globals is not None:
                data_min, data_max = feat_globals
            else:
                data_min = torch.min(observed_data[:, 0]) - 1e-3
                data_max = torch.max(observed_data[:, 0])

            normalization_parameters.append((data_min, data_max))

            normalized_observed = (observed_data[:, 0] - data_min) / (data_max - data_min)
            normalized_d = torch.zeros_like(d)
            normalized_d[observed_mask, 0] = normalized_observed
            normalized_d[observed_mask, 1] = observed_data[:, 1]
            normalized_d[missing_mask] = 0

        else:
            # Keep categorical and ordinal values unchanged
            normalized_d = d.clone()
            normalization_parameters.append((0.0, 1.0))

        normalized_data.append(normalized_d)

    return normalized_data, normalization_parameters


def y_partition(samples_y, feat_types_list, y_dim_partition):
    """
    Partitions `samples_y` according to `y_dim_partition`.

    Parameters:
    -----------
    samples_y : torch.Tensor
        The latent variable `y` tensor of shape `(batch_size, sum(y_dim_partition))`.
    
    feat_types_list : list of dict
        List of dictionaries defining variable types and dimensions.
    
    y_dim_partition : list of int
        List specifying partition sizes for `y`.

    Returns:
    --------
    list of torch.Tensor :
        A list where each entry corresponds to a partitioned segment of `samples_y`.
    """
    
    partition_indices = np.insert(np.cumsum(y_dim_partition), 0, 0)
    
    return [samples_y[:, partition_indices[i]:partition_indices[i+1]] for i in range(len(feat_types_list))]


def discrete_variables_transformation(data, types_dict):
    """
    Transforms categorical and ordinal variables into their correct numerical representations.

    Parameters:
    -----------
    data : torch.Tensor
        The dataset containing mixed-type features.
    types_dict : list of dict
        A list of dictionaries specifying the type and dimension of each feature.

    Returns:
    --------
    torch.Tensor
        A tensor where categorical variables are mapped to their indices,
        and ordinal variables are transformed using sum-based encoding.
    """

    ind_ini, output = 0, []
    for d in types_dict:
        ind_end = ind_ini + (int(d['nclass']) if d["type"] in ['cat', 'ordinal'] else int(d['dim']))
        subset = data[:, ind_ini : ind_end]  # Extract relevant columns

        if d['type'] == 'cat':
            output.append(torch.argmax(subset, dim=1, keepdim=True))  # Argmax for categorical variables
        elif d['type'] == 'ordinal':
            output.append((torch.sum(subset, dim=1, keepdim=True) - 1))  # Sum-based transformation for ordinal variables
        else:
            output.append(subset)  # Keep continuous variables unchanged
        
        ind_ini = ind_end
    
    return torch.cat(output, dim=1)


def survival_variables_transformation(data, types_dict):
    """
    Transforms categorical and ordinal variables into their correct numerical representations.

    Parameters:
    -----------
    data : torch.Tensor
        The dataset containing mixed-type features.
    types_dict : list of dict
        A list of dictionaries specifying the type and dimension of each feature.

    Returns:
    --------
    torch.Tensor
        A tensor where categorical variables are mapped to their indices,
        and ordinal variables are transformed using sum-based encoding.
    """
    output = data.clone()

    feat_idx = 0
    for d in types_dict:
        if d['type'] in ['surv','surv_weibull','surv_loglog', 'surv_piecewise']:
            subset = output[:, feat_idx : feat_idx + 2]
            time_cens = (torch.min(subset, dim=1, keepdim=True))
            output[:, feat_idx] = time_cens.values.squeeze(1)
            output[:, feat_idx + 1] = 1 - time_cens.indices.squeeze(1)
            feat_idx += 2
        else:
            feat_idx += 1
    
    return output


def encode_and_bind(df, feature):
    """
    One-hot encodes a categorical feature if it has more than 2 unique values.
    Drops the original column and appends the encoded dummies.
    
    Parameters:
        df (pd.DataFrame): The original DataFrame.
        feature (str): The feature/column name to encode.
        
    Returns:
        pd.DataFrame: Modified DataFrame with encoding applied.
    """
    unique_values = df[feature].nunique()

    if unique_values > 2:
        dummies = pd.get_dummies(df[feature], drop_first=True, prefix=feature, prefix_sep='')
        df = pd.concat([df.drop(columns=[feature]), dummies], axis=1)

    return df


def infer_rounding_step(values, coverage=0.999, rtol=1e-6, candidates=None):
    """
    Infers the coarsest grid step on which (nearly) all of `values` lie.

    Returns the largest candidate step `s` such that at least `coverage`
    fraction of the observed values fall within `rtol * s` of an integer
    multiple of `s`. The default candidate set covers powers of ten and the
    1/2, 1/4, 1/5 fractional grids (e.g. ..., 5, 2.5, 2, 1, 0.5, 0.25, 0.2,
    0.1, ...), so multiples-of-5/10 grids are detected as well.

    The `coverage` threshold is what makes this robust to a handful of
    odd values: with coverage=1.0 a couple of irregular rows force a very
    fine step, whereas coverage slightly below 1 (e.g. 0.99) treats those
    rows as noise and recovers the human-meaningful grid.

    Parameters
    ----------
    values : array-like or pandas.Series
        The column of values to inspect. NaNs/infs are ignored.
    coverage : float, default=0.999
        Minimum fraction of values that must lie on a candidate grid for
        that grid to be accepted (0 < coverage <= 1).
    rtol : float, default=1e-6
        Relative tolerance: a value `x` counts as on-grid for step `s`
        when ``abs(x/s - round(x/s)) <= rtol``.
    candidates : sequence of float or None, default=None
        Explicit candidate steps to test. When None, a default set spanning
        1e6 down to 1e-6 times {1, 1/2, 1/4, 1/5} is used. Pass your own
        (e.g. including 1/3) to encode domain-specific grids.

    Returns
    -------
    float or None
        The inferred grid step, or None when the column is empty or no
        candidate explains it (i.e. it looks continuous and should not be
        snapped).
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None

    if candidates is None:
        mantissas = [1.0, 0.5, 0.25, 0.2]                 # 1, 1/2, 1/4, 1/5
        candidates = sorted({m * 10.0 ** e                # coarsest first
                             for e in range(6, -7, -1) for m in mantissas},
                            reverse=True)

    for step in candidates:
        q = x / step
        if np.mean(np.abs(q - np.round(q)) <= rtol) >= coverage:
            return step
    return None


def infer_rounding_steps(df, cols=None, **kwargs):
    """
    Convenience wrapper applying `infer_rounding_step` to several columns.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame holding the columns to inspect.
    cols : sequence of str or None, default=None
        Column names to process. When None, every column in `df` is used.
    **kwargs
        Forwarded to `infer_rounding_step` (e.g. coverage, rtol, candidates).

    Returns
    -------
    dict
        Mapping of column name to its inferred grid step (or None when no
        step explains the column).
    """
    if cols is None:
        cols = df.columns
    return {c: infer_rounding_step(df[c], **kwargs) for c in cols}


def round_to_initial_grid(df, round_step, round_floor=None):
    """
    Snaps continuous columns of `df` onto a precision grid.

    Each column named in `round_step` is rounded to the nearest multiple of
    its step and, when a floor is supplied, clamped to that lower bound so
    rounding cannot push a value out of its valid domain (e.g. a tiny
    generated 'time' rounding down past the smallest real value). Columns
    absent from `round_step` are left untouched.

    Parameters
    ----------
    df : pandas.DataFrame
        Data to round. Not modified in place; a copy is returned.
    round_step : dict
        Mapping of column name to grid step, as produced by
        `infer_rounding_steps`.
    round_floor : dict or None, default=None
        Mapping of column name to a lower bound applied after rounding.
        When None, no clamping is performed; columns missing from the dict
        are not clamped.

    Returns
    -------
    pandas.DataFrame
        A copy of `df` with the listed columns snapped to their grid.
    """
    out = df.copy()
    for col, step in round_step.items():
        if step is None:
            continue
        out[col] = np.round(out[col] / step) * step
        if round_floor is not None and col in round_floor:
            out[col] = out[col].clip(lower=round_floor[col])
    return out

def _round_column(data_gen_round, feat_idx, series, coverage=0.99):
    """
    Infers the rounding grid of `series` and snaps
    `data_gen_round[:, feat_idx]` onto it, clipped to the series' min.
    No-op if no grid step is detected.
    """
    step = infer_rounding_step(series, coverage=coverage)
    if step is None:
        return
    round_floor = series.min()
    col = data_gen_round[:, feat_idx]
    col = np.round(col / step) * step
    data_gen_round[:, feat_idx] = col.clip(min=round_floor)


def round_data_gen(data_org, data_gen, feat_types_dict):
    """
    Rounds generated columns onto the precision grid inferred from the
    corresponding original (real) columns.

    Survival feature types occupy two tensor columns (time, event) but only
    the time column (first one) is rounded. 'pos'/'real' types occupy one
    column and are rounded directly. Any other feature type is assumed to
    occupy a single column and is left untouched.
    """
    data_gen_round = data_gen.clone()
    feat_idx = 0
    for d in feat_types_dict:
        ftype = d['type']

        if ftype in ['surv','surv_weibull','surv_loglog', 'surv_piecewise']:
            _round_column(data_gen_round, feat_idx, data_org[:, feat_idx])
            feat_idx += 2
        elif ftype in ['pos', 'real']:
            _round_column(data_gen_round, feat_idx, data_org[:, feat_idx])
            feat_idx += 1
        else:
            # Categorical / other types: nothing to round, but still
            # advance past this column.
            feat_idx += 1

    return data_gen_round

def decode_categoricals(df, cat_mapping, cols=None):
    """
    Maps categorical columns from integer codes back to their original values.

    Inverts the {original_value: integer_code} mapping produced by
    `read_data(..., return_cat_mapping=True)` and applies it, restoring
    HIVAE-decoded integer codes (0..k-1) to the dataset's original category
    labels (e.g. karnof 0/1/2/3 -> 70/80/90/100, raceth 0..3 -> 1..4).

    Parameters
    ----------
    df : pandas.DataFrame
        Data whose categorical columns hold integer codes. Not modified in
        place; a copy is returned.
    cat_mapping : dict
        Mapping of column name to a {original_value: integer_code} dict, as
        returned by `read_data` with return_cat_mapping=True.
    cols : sequence of str or None, default=None
        Subset of `cat_mapping` columns to decode. When None, every column in
        `cat_mapping` that is also present in `df` is decoded.

    Returns
    -------
    pandas.DataFrame
        A copy of `df` with the selected categorical columns mapped back to
        their original values.
    """
    out = df.copy()
    if cols is None:
        cols = [c for c in cat_mapping if c in df.columns]
    for col in cols:
        decode = {code: original for original, code in cat_mapping[col].items()}
        out[col] = out[col].round().astype(int).map(decode)
    return out



from torch.utils.data import Dataset

# class MyCustomDataset(Dataset):
#     def __init__(self, data_tensor, miss_mask_tensor):
#         self.data = data_tensor
#         self.miss = miss_mask_tensor

#     def __len__(self):
#         return self.data.shape[0]

#     def __getitem__(self, idx):
#         return self.data[idx], self.miss[idx]



class MyCustomDataset(Dataset):
    def __init__(self, data, miss_mask, types_dict):
        self.data = data
        self.miss_mask = miss_mask
        self.types_dict = types_dict
        # Precompute feature slice indices
        self.feature_slices = self._compute_feature_slices(types_dict)

    def _compute_feature_slices(self, types_dict):
        slices = []
        start = 0
        for d in types_dict:
            dim = int(d["nclass"]) if d["type"] in ['cat', 'ordinal'] else int(d["dim"])
            slices.append((start, start + dim))
            start += dim
        return slices

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = self.data[idx]
        # miss_row = self.miss_mask[idx]
        
        # Split features
        data_list = [row[start:end] for start, end in self.feature_slices]
        # miss_list = [miss_row[start:end] for start, end in self.feature_slices]

        miss_list = self.miss_mask[idx, :]
        
        return data_list, miss_list
    
