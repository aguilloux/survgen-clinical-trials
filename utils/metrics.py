#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Converted to PyTorch

Created on Mon Feb 17 20:35:11 2025

@author: Van Tuan NGUYEN
"""

import numpy as np
import pandas as pd

from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines import CoxPHFitter

from synthcity.plugins.core.dataloader import SurvivalAnalysisDataLoader
from synthcity.utils.reproducibility import clear_cache, enable_reproducible_results
from synthcity.metrics.eval import Metrics


def compute_logrank_test(control, treat):
    """
    Perform a two-sample log-rank test comparing the survival distributions
    of control and treatment groups.

    Args:
        control (DataFrame): Subset of the dataset where treatment == 0.
        treat (DataFrame): Subset of the dataset where treatment == 1.

    Returns:
        float: Negative logarithm of the p-value from the log-rank test.
    """
    surv_time_control = control['time'].values
    surv_event_control = control['censor'].values.astype(bool)
    surv_time_treat = treat['time'].values
    surv_event_treat = treat['censor'].values.astype(bool)

    result = logrank_test(
        surv_time_control, surv_time_treat,
        event_observed_A=surv_event_control,
        event_observed_B=surv_event_treat
    )
    return -np.log(result.p_value)

def log_rank(data_init, data_syn):
    """
    Evaluate the difference in survival distributions between treatment and control
    groups for both initial and synthetic datasets using the log-rank test.

    Args:
        data_init (DataFrame): Original dataset.
        data_syn (list of DataFrame): List of synthetic datasets.

    Returns:
        tuple: Log-rank test statistic for initial data and array of statistics for synthetic data.
    """
    control_init = data_init[data_init['treatment'] == 0]
    treat_init = data_init[data_init['treatment'] == 1]
    logrank_init = compute_logrank_test(control_init, treat_init)

    logrank_syn = [
        compute_logrank_test(
            data[data['treatment'] == 0],
            data[data['treatment'] == 1]
        ) for data in data_syn
    ]

    return logrank_init, np.array(logrank_syn)


def compute_multivariate_logrank_test(surv_time, treatment, surv_event, strata):
    """
    Perform a stratified log-rank test across specified strata.

    Args:
        surv_time (array): Array of survival times.
        treatment (array): Array indicating treatment group.
        surv_event (array): Event indicator array.
        strata (array): Stratification variable.

    Returns:
        float: Negative logarithm of the p-value from the stratified log-rank test.
    """
    result = multivariate_logrank_test(surv_time, treatment, surv_event, strata=strata)
    return -np.log(result.p_value)

def strata_log_rank(data_init, data_syn, strata):
    """
    Evaluate stratified survival difference between groups on initial and synthetic datasets.

    Args:
        data_init (DataFrame): Original dataset.
        data_syn (list of DataFrame): List of synthetic datasets.
        strata (str): Column name to stratify on.

    Returns:
        tuple: Stratified log-rank test statistic for initial data and array for synthetic data.
    """
    surv_time_init, surv_event_init = data_init['time'], data_init['censor'].astype(bool)
    logrank_init = compute_multivariate_logrank_test(
        surv_time_init,
        data_init['treatment'],
        surv_event_init,
        data_init[strata]
    )

    logrank_syn = [
        compute_multivariate_logrank_test(
            data['time'],
            data['treatment'],
            data['censor'].astype(bool),
            data[strata]
        ) for data in data_syn
    ]

    return logrank_init, np.array(logrank_syn)

def fit_cox_model(data, columns, strata=None):
    """
    Fit a Cox proportional hazards model optionally stratified by a variable.

    Args:
        data (DataFrame): Dataset containing survival and covariate information.
        columns (list): List of column names to include in the model.
        strata (list, optional): Stratification variable(s).

    Returns:
        tuple: Coefficients and p-values from the Cox model.
    """
    cph = CoxPHFitter()
    fit_args = {'duration_col': 'time', 'event_col': 'censor'}
    if strata:
        fit_args['strata'] = strata

    cph.fit(data[columns], **fit_args)
    return cph.summary.coef.values, cph.summary.p.values, cph.confidence_intervals_.values.flatten(), cph.standard_errors_.values.flatten()

def cox_estimation(data_init, data_syn):
    """
    Estimate Cox model coefficients and p-values for initial and synthetic datasets.

    Args:
        data_init (DataFrame): Original dataset.
        data_syn (list of DataFrame): List of synthetic datasets.

    Returns:
        tuple: Initial coefficients, synthetic coefficients, initial p-values, synthetic p-values.
    """
    columns = ['time', 'censor', 'treatment']
    coef_init, p_value_init, _, _ = fit_cox_model(data_init, columns)

    results = [fit_cox_model(data, columns) for data in data_syn]
    coef_syn, p_value_syn, _, _ = zip(*results)

    return coef_init, np.array(coef_syn), p_value_init, np.array(p_value_syn)

def strata_cox_estimation(data_init, data_syn, strata=None):
    """
    Estimate stratified Cox model coefficients and p-values for initial and synthetic datasets.

    Args:
        data_init (DataFrame): Original dataset.
        data_syn (list of DataFrame): List of synthetic datasets.
        strata (str): Column to use for stratification.

    Returns:
        tuple: Initial coefficients, synthetic coefficients, initial p-values, synthetic p-values.
    """
    columns = ['time', 'censor', 'treatment', strata]
    coef_init, p_value_init, _, _ = fit_cox_model(data_init, columns, strata=[strata])

    results = [fit_cox_model(data, columns, strata=[strata]) for data in data_syn]
    coef_syn, p_value_syn, _, _ = zip(*results)

    return coef_init, np.array(coef_syn), p_value_init, np.array(p_value_syn)

def general_metrics(data_init, data_gen, generator):
    """
    Compute a set of general quality metrics to assess synthetic survival data.

    Args:
        data_init (DataFrame): Initial real-world dataset.
        data_gen (list of DataFrame): List of generated synthetic datasets.
        generator (str): Name of the synthetic data generator.

    Returns:
        DataFrame: Summary of metric scores for each synthetic dataset.
    """

    synthcity_dataloader_init = SurvivalAnalysisDataLoader(data_init, target_column = "censor", time_to_event_column = "time")
    metrics = {
        'sanity': ['nearest_syn_neighbor_distance'],
        'stats': ['jensenshannon_dist', 'ks_test', 'survival_km_distance'],
        'performance': ['feat_rank_distance'],
        'detection': ['detection_xgb'],
        'privacy': ['k-map', 'distinct l-diversity', 'identifiability_score']
    }

    # Define expected metrics and readable names
    expected_metrics = {
        "stats.jensenshannon_dist.marginal": "J-S distance",
        "stats.ks_test.marginal": "KS test",
        "stats.survival_km_distance.abs_optimism": "Survival curves distance",
        "detection.detection_xgb.mean": "Detection XGB",
        "sanity.nearest_syn_neighbor_distance.mean": "NNDR",
        "privacy.k-map.score": "K-map score"
    }

    scores = []
    for idx, generated_data in enumerate(data_gen):
        enable_reproducible_results(idx)
        clear_cache()

        synthcity_dataloader_syn = SurvivalAnalysisDataLoader(generated_data, target_column = "censor", time_to_event_column = "time")

        # evaluation = Metrics().evaluate(X_gt=synthcity_dataloader_init, # can be dataloaders or dataframes
        #                                 X_syn=synthcity_dataloader_syn, 
        #                                 reduction='mean', # default mean
        #                                 n_histogram_bins=10, # default 10
        #                                 metrics=None, # all metrics
        #                                 task_type='survival_analysis', 
        #                                 use_cache=True)
        
        evaluation = Metrics().evaluate(X_gt=synthcity_dataloader_init, # can be dataloaders or dataframes
                                        X_syn=synthcity_dataloader_syn, 
                                        reduction='mean', # default mean
                                        n_histogram_bins=10, # default 10
                                        metrics={'stats': ['jensenshannon_dist', 'ks_test', 'survival_km_distance'], 
                                                 'detection': ['detection_xgb'],
                                                 'sanity': ['nearest_syn_neighbor_distance'],
                                                 'privacy': ['k-map']
                                                }, # compute only selected metrics
                                        task_type='survival_analysis', 
                                        # n_folds=1,
                                        use_cache=True)
        
        # selected_metrics = evaluation.T[["stats.jensenshannon_dist.marginal",
        #                                   "stats.ks_test.marginal", 
        #                                   "stats.survival_km_distance.abs_optimism",
        #                                   "detection.detection_xgb.mean", 
        #                                   "sanity.nearest_syn_neighbor_distance.mean", 
        #                                   "privacy.k-map.score"]].T["mean"].values
        # scores.append(selected_metrics)
        # print("selected_metrics: ", selected_metrics)

        # Safely retrieve all selected metrics
        values = []
        for metric in expected_metrics:
            if metric in evaluation.T.columns:
                val = evaluation.T[[metric]].T["mean"].values[0]
            else:
                val = np.nan
            values.append(val)
        # print("values: ", values)
        scores.append(values)

    score_df = pd.DataFrame(scores, columns=["J-S distance", "KS test", "Survival curves distance", 
                                             "Detection XGB", "NNDR", "K-map score"])
    score_df["generator"] = generator

    return score_df


def estimate_agreement(real_ci, augmented_est):
    """
    Args:
        real_ci (tuple): (lower, upper) bound of 95% CI from real data
        augmented_est (float): estimate from synthetic/augmented data
    Returns:
        bool: True if estimate within CI
    """
    l, u = real_ci[0], real_ci[1]
    return ((l <= augmented_est) and (augmented_est <= u)).astype(int)

def decision_agreement(init_est, init_ci, syn_est, syn_ci):
    """
    Args:
        real_est (float): real data estimate
        real_ci (tuple): (lower, upper) 95% CI real data
        aug_est (float): augmented data estimate
        aug_ci (tuple): (lower, upper) 95% CI augmented data
    Returns:
        bool: True if both have same sign and both sig/non-sig
    """
    l_init, u_init = init_ci
    l_syn, u_syn = syn_ci
    sig_real = 0 if (l_init < 0 < u_init) else 1
    sig_syn = 0 if (l_syn < 0 < u_syn) else 1

    if sig_real == 0 and sig_syn == 0:
        return 1
    elif sig_real == 1 and sig_syn == 1:
        return int(np.sign(init_est) == np.sign(syn_est))
    else:
        return 0

def standardized_difference(init_est, syn_est, init_se):
    """
    Args:
        init_est (float): real data estimate
        syn_est (float): synthetic/augmented estimate
        init_se (float): standard error from real data
    Returns:
        bool: True if difference is within ±1.96 (standard normal threshold)
    """
    z = (syn_est - init_est) / (init_se * (2 ** 0.5))  # assumes equal variance
    return (abs(z) <= 1.96).astype(int)

def ci_overlap(ci_init, ci_syn):
    """
    Args:
        ci_init, ci_syn (tuple): (lower, upper) bounds of 95% CIs
    Returns:
        float: proportion of overlap (0.0 to 1.0)
    """
    l_init, u_init = ci_init
    l_syn, u_syn = ci_syn

    overlap = max(0, min(u_init, u_syn) - max(l_init, l_syn))
    denom_init = u_init - l_init
    denom_syn = u_syn - l_syn

    # Avoid division by zero
    if denom_init == 0 or denom_syn == 0:
        return 0.0

    return 0.5 * ((overlap / denom_init) + (overlap / denom_syn))


def replicability(data_init, data_syn, generator):

    columns = ['time', 'censor', 'treatment']
    coef_init, _, ci_init, se_init = fit_cox_model(data_init, columns)

    results = [fit_cox_model(data, columns) for data in data_syn]
    coef_syn, _, _, se_syn = zip(*results)
    max_len_samples = len(data_syn)
    list_len_samples = np.arange(int(.2 * max_len_samples), max_len_samples, int(.2 * max_len_samples)).tolist()
    if max_len_samples not in list_len_samples:
        list_len_samples += [max_len_samples]
    score_df = pd.DataFrame(columns=["Generator", "Nb generated datasets", "Estimate agreement", "Decision agreement", "Standardized difference", "CI overlap"])
    for m in list_len_samples:
        coef_syn_, se_syn_ = np.array(coef_syn)[:m], np.array(se_syn)[:m]
        coef_syn_mean = coef_syn_.mean()
        var_syn_mean = (se_syn_**2).mean()
        # imputation_var_syn = (1 / (len(coef_syn) - 1)) * np.sum([(coef_syn_ - coef_syn_mean)**2 for coef_syn_ in coef_syn])
        # adjusted_var_syn = (imputation_var_syn / len(coef_syn)) + var_syn_mean
        adjusted_var_syn = (1/m + 1) * var_syn_mean
        ci_syn = (coef_syn_mean - 1.96 * np.sqrt(adjusted_var_syn), coef_syn_mean + 1.96 * np.sqrt(adjusted_var_syn))

        res = [estimate_agreement(ci_init, coef_syn_mean),
            decision_agreement(coef_init[0], ci_init, coef_syn_mean, ci_syn),
            standardized_difference(coef_init[0], coef_syn_mean, se_init[0]),
            ci_overlap(ci_init, ci_syn)]

        score_df.loc[len(score_df)] = [generator, m] + res

    return score_df

def replicability_ext(data_init, data_syn, generator):

    columns = ['time', 'censor', 'treatment']
    coef_init, _, ci_init, se_init = fit_cox_model(data_init, columns)

    results = [fit_cox_model(data, columns) for data in data_syn]
    coef_syn, _, _, se_syn = zip(*results)
    max_len_samples = len(data_syn)
    list_len_samples = np.arange(int(.2 * max_len_samples), max_len_samples, int(.2 * max_len_samples)).tolist()
    if max_len_samples not in list_len_samples:
        list_len_samples += [max_len_samples]
    score_df = pd.DataFrame(columns=["Generator", "Nb generated datasets", "Estimate agreement", "Decision agreement", "Standardized difference", "CI overlap"])
    for m in list_len_samples:
        res = []
        for n in range(m):
            coef_syn_, se_syn_ = np.array(coef_syn)[n][0], np.array(se_syn)[n][0]
            ci_syn = (coef_syn_ - 1.96 * se_syn_, coef_syn_ + 1.96 * se_syn_)

            res.append([estimate_agreement(ci_init, coef_syn_),
                    decision_agreement(coef_init[0], ci_init, coef_syn_, ci_syn),
                    standardized_difference(coef_init[0], coef_syn_, se_init[0]),
                    ci_overlap(ci_init, ci_syn)])
        score_df.loc[len(score_df)] = [generator, m] + np.array(res).mean(axis=0).tolist()

    return score_df