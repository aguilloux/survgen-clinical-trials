#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Converted to PyTorch

Created on Mon Feb 17 20:35:11 2025

@author: Van Tuan NGUYEN
"""

import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines import CoxPHFitter
from tableone import TableOne

from synthcity.plugins.core.dataloader import SurvivalAnalysisDataLoader, GenericDataLoader
from synthcity.utils.reproducibility import clear_cache, enable_reproducible_results
from synthcity.metrics.eval import Metrics
from synthcity.metrics.eval_privacy import DomiasMIAKDE, DomiasMIAPrior, DomiasMIABNAF


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

def log_rank(data_init, data_syn, interest_var='treatment'):
    """
    Evaluate the difference in survival distributions between two groups
    defined by a binary covariate (`interest_var`), for both the initial
    and the synthetic datasets, using the log-rank test.

    Args:
        data_init (DataFrame): Original dataset.
        data_syn (list of DataFrame): List of synthetic datasets.
        interest_var (str): Name of the binary column used to split the
            data into the two groups (values 0 and 1). Defaults to
            'treatment'; any binary covariate present in the dataframes
            can be passed.

    Returns:
        tuple: Log-rank test statistic for initial data and array of statistics for synthetic data.
    """
    control_init = data_init[data_init[interest_var] == 0]
    treat_init = data_init[data_init[interest_var] == 1]
    logrank_init = compute_logrank_test(control_init, treat_init)

    logrank_syn = [
        compute_logrank_test(
            data[data[interest_var] == 0],
            data[data[interest_var] == 1]
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

def compute_nndr(data_real, data_syn, columns=None, eps=1e-8):
    """
    Per-record Nearest Neighbor Distance Ratio (NNDR) in the synthetic -> real direction.

    For each synthetic record, find its nearest (d1) and second-nearest (d2) records
    in the real dataset and return d1 / d2. Features are standardized using statistics
    fitted on the real data, so a wide-scale column (e.g. 'time') cannot dominate the
    Euclidean distance. Because the query set (synthetic) differs from the search set
    (real), d1 is a genuine real neighbor, not a trivial self-match.

    The ratio lies in [0, 1]: values near 0 flag synthetic records that single out one
    real record (potential privacy leakage), values near 1 indicate records sitting in a
    dense region without pinpointing any single individual.

    Args:
        data_real (DataFrame): Real (reference) dataset; the neighbor search space.
        data_syn (DataFrame): Synthetic dataset; the query records.
        columns (list, optional): Columns to use for the distance. Defaults to the
            numeric columns shared by both frames.
        eps (float): Floor on d2 to avoid division by zero when a synthetic record
            coincides with >=2 real records (d1 = d2 = 0 -> NNDR = 0, i.e. flagged risky).

    Returns:
        np.ndarray: NNDR value in [0, 1] for each synthetic record.
    """
    real = pd.DataFrame(data_real).reset_index(drop=True)
    syn = pd.DataFrame(data_syn).reset_index(drop=True)

    if columns is None:
        columns = [c for c in real.columns
                   if c in syn.columns and np.issubdtype(real[c].dtype, np.number)]
    if not columns:
        raise ValueError("No shared numeric columns to compute NNDR on.")

    # Fit the standardization on the real data: it is the reference frame the synthetic
    # records are scored against. StandardScaler maps zero-variance columns to scale 1,
    # so constant columns stay finite instead of producing NaNs.
    scaler = StandardScaler().fit(real[columns].to_numpy())
    real_scaled = scaler.transform(real[columns].to_numpy())
    syn_scaled = scaler.transform(syn[columns].to_numpy())

    nn = NearestNeighbors(n_neighbors=2).fit(real_scaled)
    dist, _ = nn.kneighbors(syn_scaled)  # query synthetic against real
    d1, d2 = dist[:, 0], dist[:, 1]
    return d1 / np.maximum(d2, eps)

def nndr(data_real, data_syn, columns=None, percentile=5, eps=1e-8):
    """
    Reduced NNDR statistics for the synthetic -> real direction.

    Args:
        data_real (DataFrame): Real (reference) dataset.
        data_syn (DataFrame): Synthetic dataset.
        columns (list, optional): Columns to use. Defaults to shared numeric columns.
        percentile (float): Low percentile summarizing the risky tail. Defaults to 5.
        eps (float): Division-by-zero floor passed to compute_nndr.

    Returns:
        dict: {"mean": float, "p{percentile}": float}. The mean matches the homogeneous
            "everything is a mean" style of the general_metrics tables; the low percentile
            captures the risky tail (synthetic records with NNDR near 0).
    """
    scores = compute_nndr(data_real, data_syn, columns=columns, eps=eps)
    return {
        "mean": float(np.mean(scores)),
        f"p{percentile}": float(np.percentile(scores, percentile)),
    }

def general_metrics(data_init, data_gen, generator, include_nndr=True):
    """
    Compute a set of general quality metrics to assess synthetic survival data.

    Args:
        data_init (DataFrame): Initial real-world dataset.
        data_gen (list of DataFrame): List of generated synthetic datasets.
        generator (str): Name of the synthetic data generator.
        include_nndr (bool): Append a mean NNDR (synthetic -> real) column. Defaults to True.

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
        "sanity.nearest_syn_neighbor_distance.mean": "NSND",
        "privacy.k-map.score": "K-map score",
        "privacy.identifiability_score.score": "Identifiability score"
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
                                                 'privacy': ['k-map', 'identifiability_score']
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
        if include_nndr:
            values.append(nndr(data_init, generated_data)["mean"])
        # print("values: ", values)
        scores.append(values)

    columns = ["J-S distance", "KS test", "Survival curves distance",
               "Detection XGB", "NSND", "K-map score", "Identifiability score"]
    if include_nndr:
        columns = columns + ["NNDR"]
    score_df = pd.DataFrame(scores, columns=columns)
    score_df["generator"] = generator

    return score_df

def general_metrics_modular(data_init, data_gen, generator, metrics = {
        'sanity': ['nearest_syn_neighbor_distance'],
        'stats': ['jensenshannon_dist', 'ks_test', 'survival_km_distance'],
        'performance': ['feat_rank_distance'],
        'detection': ['detection_xgb'],
        'privacy': ['k-map', 'distinct l-diversity', 'identifiability_score']
    }, include_nndr=True, include_tableone_min_p_value=False, categorical=None,
    continuous=None, nonnormal=None):
    """
    Compute a configurable set of quality metrics to assess synthetic survival data.

    Unlike general_metrics, this function accepts a custom metrics dict, allowing
    callers to select which synthcity metric groups and sub-metrics to evaluate.
    Only metrics present in the expected_metrics mapping are included in the output;
    any requested metric not found in the evaluation result is silently skipped.

    Args:
        data_init (DataFrame): Initial real-world dataset.
        data_gen (list of DataFrame): List of generated synthetic datasets.
        generator (str): Name of the synthetic data generator.
        metrics (dict, optional): Dictionary mapping synthcity metric categories to
            lists of metric names to compute. Defaults to a standard set covering
            stats, detection, sanity, and privacy metrics.
        include_nndr (bool): Append a mean NNDR (synthetic -> real) column. Defaults to True.
        include_tableone_min_p_value (bool): Append a "TableOne min p-value" column holding
            the smallest per-variable p-value from a TableOne real-vs-synthetic comparison
            (plus the survival log-rank p-value). A high value means no single variable
            distinguishes synthetic from real. Requires `categorical` and `continuous`.
            Defaults to False.
        categorical (list, optional): Categorical column names for TableOne. Required when
            include_tableone_min_p_value is True.
        continuous (list, optional): Continuous column names for TableOne. Required when
            include_tableone_min_p_value is True.
        nonnormal (list, optional): Subset of continuous columns to summarize/test
            non-parametrically in TableOne. Defaults to None (all treated as normal).

    Returns:
        DataFrame: Summary of metric scores for each synthetic dataset, with one
            column per matched metric and a 'generator' column.
    """

    if include_tableone_min_p_value:
        if (categorical is None) or (continuous is None):
            raise ValueError(
                "include_tableone_min_p_value=True requires both `categorical` and "
                "`continuous` to be provided (lists of column names for TableOne)."
            )
        # data_init is the reference frame for every generated dataset; tag it once.
        df_init_tableone = data_init.copy()
        df_init_tableone["sample"] = 1
    synthcity_dataloader_init = SurvivalAnalysisDataLoader(data_init, target_column = "censor", time_to_event_column = "time")

    # Map synthcity output keys to human-readable column names
    expected_metrics = {
        "stats.jensenshannon_dist.marginal": "J-S distance",
        "stats.ks_test.marginal": "KS test",
        "stats.survival_km_distance.abs_optimism": "Survival curves distance",
        "detection.detection_xgb.mean": "Detection XGB",
        "sanity.nearest_syn_neighbor_distance.mean": "NSND",
        "privacy.k-map.score": "K-map score",
        "privacy.identifiability_score.score": "Identifiability score"
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
                                        metrics=metrics, # compute only selected metrics
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
        name_columns = []
        for metric in expected_metrics:
            if metric in evaluation.T.columns:
                val = evaluation.T[[metric]].T["mean"].values[0]
                values.append(val)
                name_columns.append(expected_metrics[metric])
        if include_nndr:
            values.append(nndr(data_init, generated_data)["mean"])
            name_columns.append("NNDR")
        if include_tableone_min_p_value:
            # Tag the synthetic frame and compare it against the real one with TableOne,
            # grouping on `sample` (1 = real, 0 = synthetic). The smallest p-value across
            # all variable tests (incl. the survival log-rank) is the hardest-to-pass test.
            df_gen_tableone = generated_data.copy()
            df_gen_tableone["sample"] = 0
            _, min_p_value, _, _ = tableone_tests(
                pd.concat([df_init_tableone, df_gen_tableone], ignore_index=True),
                groupby="sample",
                categorical=categorical,
                continuous=continuous,
                nonnormal=nonnormal,
            )
            values.append(min_p_value)
            name_columns.append("TableOne min p-value")
        # print("values: ", values)
        scores.append(values)

    score_df = pd.DataFrame(scores, columns=name_columns)
    score_df["generator"] = generator

    return score_df

def tableone_tests(df, groupby, categorical, continuous, nonnormal):
    table1_sel = TableOne(df, groupby=groupby, categorical=categorical, continuous=continuous, nonnormal=nonnormal, pval=True).tableone

    # Remove the sample variable from the TableOne consideration (since it is our groupby variable)
    table1_sel = table1_sel[~table1_sel.index.isin([('sample, n (%)', '0'), ('sample, n (%)', '1')])]

    # Select the relevant part of the DataFrame
    table1_sel = table1_sel[f'Grouped by {groupby}']
    table1_sel = table1_sel[~table1_sel['P-Value'].isin([''])]

    # For p-values under 0.001, replace them by 0 in order to process it as a numerical variable
    table1_sel.loc[(table1_sel['P-Value'] == '<0.001'), 'P-Value'] = '0.0'

    # Find the lowest p-value and the column(s) associated to it
    min_p_value = table1_sel['P-Value'].astype('float32').min()
    cols_min_p_value = [index_i[0].split(',')[0] for index_i in list(table1_sel.loc[table1_sel['P-Value'].astype('float32') == min_p_value].index)]

    # Compute the sum of all the p-values present in the tableone
    sum_p_values = table1_sel['P-Value'].astype('float32').sum()

    if groupby in ['sample', 'treatment']:
        p_value_surv = np.exp(-compute_logrank_test(df[df[groupby] == 1], df[df[groupby] == 0]))
        sum_p_values += p_value_surv
        if p_value_surv < min_p_value:
            min_p_value = p_value_surv
            cols_min_p_value = ['survcens']
        elif p_value_surv == min_p_value:
            cols_min_p_value.append('survcens')
        
        survcens_row = {}
        for table1_column in list(table1_sel.columns):
            survcens_row[table1_column] = ''
        survcens_row['P-Value'] = str(np.round(p_value_surv, 3))
        table1_sel = pd.concat([table1_sel, pd.DataFrame(survcens_row, index=('survcens', ''))]).iloc[:-1]
        
    
    return table1_sel, min_p_value, cols_min_p_value, sum_p_values

def variable_pct_passed_tests(data_init, data_gen, categorical, continuous, nonnormal):
    df_init_control_ext = data_init.copy()
    df_init_control_ext['sample'] = 1

    variables_pct_failed_tests = {}
    for variable in categorical+continuous+['survcens']:
        variables_pct_failed_tests[variable] = []

    for generated_data in data_gen:
        df_gen_control_ext = generated_data.copy()
        df_gen_control_ext['sample'] = 0

        table1, _, _, _, = tableone_tests(pd.concat([df_init_control_ext, df_gen_control_ext], ignore_index=True),
                                          groupby='sample', categorical=categorical, continuous=continuous,
                                          nonnormal=nonnormal)
        for i in range(len(table1)):
            name_str = table1.iloc[i].name
            if name_str != 'survcens':
                variable_name = name_str[0].split(',')[0]
            else:
                variable_name = name_str
            significant_difference = 1 if float(table1.iloc[i]['P-Value']) > 0.05 else 0
            variables_pct_failed_tests[variable_name].append(significant_difference)

    for variable in categorical+continuous+['survcens']:
        variables_pct_failed_tests[variable] = np.array(variables_pct_failed_tests[variable]).mean()
    return variables_pct_failed_tests

_DOMIAS_VARIANTS = {
    "KDE": DomiasMIAKDE,
    "prior": DomiasMIAPrior,
    "BNAF": DomiasMIABNAF,
}


def _run_domias_attack(variant, members, nonmembers, syn, ref_syn, reference_size, random_state, workspace):
    """
    Instantiate the requested DOMIAS evaluator and run it on a single, already
    column-aligned split.

    Calls the protected ``_evaluate`` directly instead of the public ``evaluate``: the latter
    caches results keyed only on the hashes of ``X_gt`` and ``X_syn`` (see
    ``PrivacyEvaluator.evaluate``), ignoring ``X_train`` and ``reference_size``, so across folds
    it would silently return stale scores.

    Returns:
        dict: synthcity's raw DOMIAS output, with keys ``"accuracy"`` and ``"aucroc"``.
    """
    evaluator = _DOMIAS_VARIANTS[variant](
        reduction="mean",
        use_cache=False,
        random_state=random_state,
        workspace=workspace,
    )
    return evaluator._evaluate(
        GenericDataLoader(nonmembers),   # X_gt          -> non-members (first k) + reference set (last k)
        GenericDataLoader(syn),          # synth_set     -> synthetic sample
        GenericDataLoader(members),      # X_train       -> members
        GenericDataLoader(ref_syn),      # synth_val_set -> only used by BNAF
        reference_size=reference_size,
    )


def membership_inference_attack(
    data_members,
    data_nonmembers,
    data_syn,
    data_ref_syn=None,
    variant="KDE",
    reference_size=30,
    drop_constant=True,
    fallback_to_bnaf=True,
    random_state=0,
    workspace=None,
):
    """
    Run a DOMIAS membership inference attack (MIA) against a synthetic dataset.

    DOMIAS (van Breugel et al., AISTATS 2023) detects generator overfitting by comparing,
    for each real record, a density ratio p_synthetic(x) / p_reference(x). Records the
    generator memorised (members) tend to score higher than fresh held-out records
    (non-members); the attack's ability to separate the two is reported as an AUCROC.
    AUCROC ~ 0.5 means the attacker cannot tell members from non-members (good privacy);
    AUCROC -> 1.0 indicates membership leakage.

    The attack needs three pieces of data that a plain real-vs-synthetic comparison
    (e.g. general_metrics_modular) does not provide, which is why this lives in its own
    function:
        - members      : real rows the generator was trained on,
        - non-members  : real rows held out from training (also sliced to build the
                         reference set used for density estimation),
        - synthetic    : data produced by the generator trained on ``members``.

    Density estimator variants (``variant``):
        - "KDE"   : gaussian_kde on the synthetic sample and on the reference slice. Fast,
                    but gaussian_kde raises on singular covariance, which happens when a
                    column is (near-)constant (e.g. an all-censored slice, or a collapsed
                    binary column in the synthetic data).
        - "prior" : gaussian_kde on the synthetic sample, analytic Gaussian as reference.
                    Fast; shares the synthetic-side fragility of KDE.
        - "BNAF"  : a normalizing-flow neural density estimator. Robust to degenerate /
                    low-dimensional data, but ~100-1000x slower (it trains two small networks
                    per call). Use this to run everything on BNAF regardless of column shape.

    Robustness handling for the fast variants:
        - ``drop_constant=True`` removes any non-continuous column that is constant in the data
          gaussian_kde fits on (the synthetic sample and the reference slice). A constant column
          carries no membership signal, so dropping it is information-free. Columns that are
          (quasi-)continuous over the real data are never dropped (the reference density needs
          at least one), so a degenerate continuous column falls through to the BNAF fallback.
        - ``fallback_to_bnaf=True`` retries with BNAF if KDE/prior still fail numerically.

    Args:
        data_members (DataFrame): Real rows used to train the generator (the "members").
        data_nonmembers (DataFrame): Held-out real rows (the "non-members"). Should contain at
            least ``2 * reference_size`` rows: the first ``reference_size`` become test
            non-members and the last ``reference_size`` become the density reference set, so the
            two slices stay disjoint. ``reference_size`` is capped automatically (with a warning)
            if there are too few rows.
        data_syn (DataFrame): Synthetic data from the generator trained on ``data_members``.
        data_ref_syn (DataFrame, optional): Reference/validation synthetic set used by BNAF to
            fit the synthetic density. Ignored by KDE/prior. Defaults to ``data_syn``.
        variant (str): "KDE" (default), "prior", or "BNAF".
        reference_size (int): Size of each held-out slice (non-members and reference set).
            Defaults to 30. Larger values give a more reliable AUCROC and make accidental
            constant slices less likely, at the cost of a larger held-out set.
        drop_constant (bool): Drop constant columns before density estimation. Defaults to True.
        fallback_to_bnaf (bool): Retry with BNAF if KDE/prior fail numerically. Defaults to True.
        random_state (int): Seed for the evaluator. Defaults to 0.
        workspace (str or Path, optional): Scratch directory for synthcity. Defaults to a
            temporary directory.

    Returns:
        dict: {
            "aucroc": float,                    # primary MIA score (0.5 = no leakage)
            "accuracy": float,
            "variant_used": str,                # differs from ``variant`` if the BNAF fallback fired
            "reference_size": int,              # possibly capped
            "n_features_used": int,             # after the constant-column guard
            "dropped_constant_columns": list,   # column names removed by the guard
        }

    Note:
        Columns must be numeric and shared across all inputs; only the common columns (in
        ``data_members`` order) are used. All available members are tested against
        ``reference_size`` non-members, so the test set is intentionally imbalanced -- average
        the AUCROC over several synthetic draws / cross-validation folds for a stable estimate.
    """
    if variant not in _DOMIAS_VARIANTS:
        raise ValueError(
            f"Unknown variant {variant!r}. Choose from {sorted(_DOMIAS_VARIANTS)}."
        )

    members = pd.DataFrame(data_members).reset_index(drop=True)
    nonmembers = pd.DataFrame(data_nonmembers).reset_index(drop=True)
    syn = pd.DataFrame(data_syn).reset_index(drop=True)
    using_ref_syn = data_ref_syn is not None
    ref_syn = pd.DataFrame(data_ref_syn).reset_index(drop=True) if using_ref_syn else syn

    # DOMIAS concatenates these frames, so they must expose identical columns in identical
    # order. Keep only the columns shared by every frame, in ``members`` order.
    frames = [members, nonmembers, syn, ref_syn]
    common = [c for c in members.columns if all(c in f.columns for f in frames)]
    if not common:
        raise ValueError(
            "No shared feature columns across members / non-members / synthetic data."
        )
    members, nonmembers, syn, ref_syn = members[common], nonmembers[common], syn[common], ref_syn[common]

    # DOMIAS' reference density (normal_func_feat) needs at least one (quasi-)continuous feature
    # (>=10 unique values pooled over the real data); otherwise synthcity raises internally.
    pooled_unique = pd.concat([members, nonmembers], ignore_index=True).nunique()
    continuous_cols = {c for c in common if pooled_unique[c] >= 10}
    if not continuous_cols:
        raise ValueError(
            "DOMIAS requires at least one (quasi-)continuous feature (>=10 unique values); "
            "none of the shared columns qualify."
        )

    # Holdout budget: non-members and the reference set are disjoint slices of the held-out
    # rows, so we need 2 * reference_size of them.
    max_reference = len(nonmembers) // 2
    if max_reference < 1:
        raise ValueError(
            f"Need at least 2 non-member rows to form the test/reference slices; got {len(nonmembers)}."
        )
    if reference_size > max_reference:
        warnings.warn(
            f"reference_size={reference_size} is too large for {len(nonmembers)} non-member rows; "
            f"capping to {max_reference} so the non-member and reference slices stay disjoint."
        )
        reference_size = max_reference

    # Constant-column guard: gaussian_kde (KDE/prior) inverts the covariance of its inputs and
    # blows up on a constant column. Drop any non-continuous column that is constant in a frame
    # gaussian_kde fits on -- the synthetic sample and the reference slice (the last
    # reference_size held-out rows). Continuous columns are protected so the reference density
    # keeps a feature; a degenerate continuous column instead trips the BNAF fallback below.
    dropped = []
    if drop_constant:
        guard_frames = [syn, nonmembers.iloc[-reference_size:]]
        constant = sorted(
            c for c in common
            if c not in continuous_cols
            and any(f[c].nunique(dropna=False) <= 1 for f in guard_frames)
        )
        keep = [c for c in common if c not in constant]
        if constant and keep:
            dropped = constant
            common = keep
            members, nonmembers, syn, ref_syn = members[keep], nonmembers[keep], syn[keep], ref_syn[keep]

    workspace = Path(workspace) if workspace is not None else Path(tempfile.gettempdir()) / "synthcity_mia_workspace"

    variant_used = variant
    try:
        result = _run_domias_attack(
            variant, members, nonmembers, syn, ref_syn, reference_size, random_state, workspace
        )
    except (LinAlgError, ValueError) as err:
        if variant in ("KDE", "prior") and fallback_to_bnaf:
            warnings.warn(
                f"DOMIAS {variant} failed numerically ({type(err).__name__}: {err}); "
                f"falling back to BNAF."
            )
            variant_used = "BNAF"
            result = _run_domias_attack(
                "BNAF", members, nonmembers, syn, ref_syn, reference_size, random_state, workspace
            )
        else:
            raise

    return {
        "aucroc": float(result["aucroc"]),
        "accuracy": float(result["accuracy"]),
        "variant_used": variant_used,
        "reference_size": int(reference_size),
        "n_features_used": len(common),
        "dropped_constant_columns": dropped,
    }


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


def map_metrics_HPO(metrics_list):
    expected_metrics = {
            "stats.jensenshannon_dist.marginal": "min",
            "stats.ks_test.marginal": "max",
            "stats.survival_km_distance.abs_optimism": "min",
            "detection.detection_xgb.mean": "min",
            "sanity.nearest_syn_neighbor_distance.mean": "max",
            "privacy.k-map.score": "max",
            "privacy.identifiability_score.score": "min"
        }
    short_to_full = {k.split(".")[1]: k for k in expected_metrics}
    metrics_synthcity = []
    for metric in metrics_list:
        if metric not in short_to_full:
            raise ValueError(f"Unexpected metric: {metric}. Expected one of: {list(short_to_full.keys())}")
        metrics_synthcity.append(short_to_full[metric])
    metrics_dict_evaluation = {}
    for metric in metrics_synthcity:
        if metric.startswith("stats."):
            metrics_dict_evaluation.setdefault("stats", []).append(metric.split(".")[1])
        elif metric.startswith("detection."):
            metrics_dict_evaluation.setdefault("detection", []).append(metric.split(".")[1])
        elif metric.startswith("sanity."):
            metrics_dict_evaluation.setdefault("sanity", []).append(metric.split(".")[1])
        elif metric.startswith("privacy."):
            metrics_dict_evaluation.setdefault("privacy", []).append(metric.split(".")[1])
    return metrics_dict_evaluation, metrics_synthcity, expected_metrics