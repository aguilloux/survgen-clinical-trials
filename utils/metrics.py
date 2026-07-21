#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Converted to PyTorch

Created on Mon Feb 17 20:35:11 2025

@author: Van Tuan NGUYEN
"""

import warnings

import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines import CoxPHFitter
from tableone import TableOne

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

def compute_nndr(data_real, data_syn, columns=None, categorical=None, eps=1e-8):
    """
    Per-record Nearest Neighbor Distance Ratio (NNDR) in the synthetic -> real direction.

    For each synthetic record, find its nearest (d1) and second-nearest (d2) records
    in the real dataset and return d1 / d2. Continuous features are standardized using
    statistics fitted on the real data, so a wide-scale column (e.g. 'time') cannot
    dominate the Euclidean distance. Categorical features (listed in ``categorical``) are
    one-hot encoded instead, so an integer code 0..k never imposes a spurious
    0 < 1 < ... < k ordering and every category mismatch carries the same fixed weight
    regardless of cardinality or level frequency. Because the query set (synthetic)
    differs from the search set (real), d1 is a genuine real neighbor, not a trivial
    self-match.

    The ratio lies in [0, 1]: values near 0 flag synthetic records that single out one
    real record (potential privacy leakage), values near 1 indicate records sitting in a
    dense region without pinpointing any single individual.

    Args:
        data_real (DataFrame): Real (reference) dataset; the neighbor search space.
        data_syn (DataFrame): Synthetic dataset; the query records.
        columns (list, optional): Columns to use for the distance. Defaults to the
            numeric columns shared by both frames, plus any columns named in
            ``categorical``.
        categorical (list, optional): Subset of ``columns`` to treat as categorical --
            one-hot encoded (real and synthetic encoded together so they share dummy
            columns) rather than standardized. Defaults to None (every column continuous).
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
        # Keep explicitly-declared categoricals even if their (encoded) dtype is non-numeric.
        for c in (categorical or []):
            if c in real.columns and c in syn.columns and c not in columns:
                columns.append(c)
    if not columns:
        raise ValueError("No shared numeric columns to compute NNDR on.")

    categorical = [c for c in (categorical or []) if c in columns]
    continuous = [c for c in columns if c not in categorical]

    # Continuous block: standardized on the real data (the reference frame the synthetic
    # records are scored against). StandardScaler maps zero-variance columns to scale 1,
    # so constant columns stay finite instead of producing NaNs. Categorical block: one-hot
    # on real+synthetic together (identical dummy columns), left as 0/1 so every category
    # mismatch adds the same fixed distance -- no false ordinality, no cardinality blow-up.
    real_blocks, syn_blocks = [], []

    if continuous:
        scaler = StandardScaler().fit(real[continuous].to_numpy())
        real_blocks.append(scaler.transform(real[continuous].to_numpy()))
        syn_blocks.append(scaler.transform(syn[continuous].to_numpy()))

    if categorical:
        dummies = pd.get_dummies(
            pd.concat([real[categorical], syn[categorical]], ignore_index=True),
            columns=categorical,
        )
        real_blocks.append(dummies.iloc[:len(real)].to_numpy(dtype=float))
        syn_blocks.append(dummies.iloc[len(real):].to_numpy(dtype=float))

    real_scaled = np.hstack(real_blocks)
    syn_scaled = np.hstack(syn_blocks)

    nn = NearestNeighbors(n_neighbors=2).fit(real_scaled)
    dist, _ = nn.kneighbors(syn_scaled)  # query synthetic against real
    d1, d2 = dist[:, 0], dist[:, 1]
    return d1 / np.maximum(d2, eps)

def nndr(data_real, data_syn, columns=None, categorical=None, percentile=5, eps=1e-8):
    """
    Reduced NNDR statistics for the synthetic -> real direction.

    Args:
        data_real (DataFrame): Real (reference) dataset.
        data_syn (DataFrame): Synthetic dataset.
        columns (list, optional): Columns to use. Defaults to shared numeric columns.
        categorical (list, optional): Columns to one-hot encode instead of standardize,
            forwarded to compute_nndr. Defaults to None.
        percentile (float): Low percentile summarizing the risky tail. Defaults to 5.
        eps (float): Division-by-zero floor passed to compute_nndr.

    Returns:
        dict: {"mean": float, "p{percentile}": float}. The mean matches the homogeneous
            "everything is a mean" style of the general_metrics tables; the low percentile
            captures the risky tail (synthetic records with NNDR near 0).
    """
    scores = compute_nndr(data_real, data_syn, columns=columns, categorical=categorical, eps=eps)
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
            When `categorical` is given, those columns are one-hot encoded in the NNDR
            distance (otherwise every column is treated as continuous).
        include_tableone_min_p_value (bool): Append a "TableOne min p-value" column holding
            the smallest per-variable p-value from a TableOne real-vs-synthetic comparison
            (plus the survival log-rank p-value). A high value means no single variable
            distinguishes synthetic from real. Requires `categorical` and `continuous`.
            Defaults to False.
        categorical (list, optional): Categorical column names. Required when
            include_tableone_min_p_value is True (used by TableOne). Also used by NNDR when
            include_nndr is True -- forwarded to compute_nndr so these columns are one-hot
            encoded rather than treated as continuous. Optional for NNDR: omitting it just
            falls back to all-continuous distances.
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
            values.append(nndr(data_init, generated_data, categorical=categorical)["mean"])
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

def membership_inference_attack(
    data_members,
    data_nonmembers,
    data_syn,
    columns=None,
    categorical=None,
    bootstrap=False,
    n_bootstrap=1000,
    random_state=1,
    tpr_at_fpr=(0.1, 0.2),
    return_boot_auc=False,
):
    """
    GAN-Leaks full black-box membership inference attack (MIA) against synthetic data.

    Scores each real record by its distance to the nearest synthetic record (closer => more
    likely a training member) and reports how well that score separates known members
    (``data_members``, the generator's training rows) from non-members (``data_nonmembers``, a
    same-distribution holdout the generator never saw), as an ROC AUC. AUC ~ 0.5 means members
    and non-members are indistinguishable (no membership leakage); AUC -> 1.0 means the generator
    places synthetic mass specifically near the records it was trained on (memorisation).

    Because the score is a members-vs-holdout *contrast*, the generator's overall fidelity
    cancels (it shifts both groups' distances together): only the train/holdout asymmetry --
    memorisation -- moves the AUC. That is what distinguishes a real MIA from NNDR /
    identifiability, which read a raw distance and so blend fidelity into the value.

    Continuous columns are standardised on the members; categorical columns (``categorical``)
    are one-hot encoded across members + non-members + synthetic together, matching
    compute_nndr's mixed-type handling (a k-level code imposes no false 0<1<...<k ordering).

    Args:
        data_members (DataFrame): Real rows the generator was trained on (members).
        data_nonmembers (DataFrame): Same-distribution held-out rows (non-members) -- a fresh
            simulation draw for simulated data, or a train/holdout split for real data.
        data_syn (DataFrame or list of DataFrame): Synthetic data. A list (e.g. the Monte-Carlo
            replicates) is pooled into one reference set: membership is a property of the trained
            model, so the attack uses all the generation at once rather than scoring each replicate.
        columns (list, optional): Columns to use. Defaults to the numeric columns shared by all
            inputs, plus any named in ``categorical``.
        categorical (list, optional): Columns to treat as categorical (one-hot encoded).
        bootstrap (bool): Add a bootstrap confidence interval for the AUC. Defaults to False.
        n_bootstrap (int): Bootstrap resamples when ``bootstrap=True``. Defaults to 1000.
        random_state (int): Seed for the bootstrap. Defaults to 0.
        tpr_at_fpr (float, iterable of float, or None): FPR operating point(s) at which to also
            report the true-positive rate (TPR@FPR), a worst-case / tail-leakage metric that AUC
            averages away. Defaults to (0.1, 0.2). Set to None to skip. Each target's resolution
            is capped by the non-member count: the finest usable FPR is ~1 / n_nonmembers, and a
            target of ``f`` rests on about ``f * n_nonmembers`` holdout records -- keep at least
            ~15-20 there (e.g. FPR >= 0.1 at ~180 non-members), or the estimate is single-record
            noise. Always read the TPR@FPR values with their bootstrap CIs.
        return_boot_auc (bool): When ``bootstrap=True``, also return the full vector of the
            ``n_bootstrap`` resampled AUCs (not just its CI bounds), so the caller can inspect or
            re-aggregate the bootstrap distribution. Ignored when ``bootstrap=False``. Defaults
            to False.

    Returns:
        dict: {"mia_auc": float, "n_members": int, "n_nonmembers": int}, plus one
            "tpr@fpr=<f>" entry per requested ``tpr_at_fpr`` target, and, when ``bootstrap=True``,
            also {"ci_low": float, "ci_high": float} for the AUC and
            "tpr@fpr=<f>_ci_low"/"tpr@fpr=<f>_ci_high" for each target. When additionally
            ``return_boot_auc=True``, a {"boot_auc": np.ndarray} entry holding the whole vector of
            resampled AUCs.

    Note:
        The AUC's precision is set by the number of member / non-member records, not by how much
        synthetic you pool -- at ~100 non-members the 95% CI is roughly +/-0.07, so treat only
        AUC >~ 0.6 as evidence of leakage. As a positive control, run it on reconstruction-based
        (posterior) generation, where it should read clearly above 0.5.
    """
    members = pd.DataFrame(data_members).reset_index(drop=True)
    nonmembers = pd.DataFrame(data_nonmembers).reset_index(drop=True)
    if isinstance(data_syn, (list, tuple)):
        syn = pd.concat([pd.DataFrame(s) for s in data_syn], ignore_index=True)
    else:
        syn = pd.DataFrame(data_syn).reset_index(drop=True)

    # Columns: numeric columns shared by all three frames, plus any declared categoricals.
    frames = [members, nonmembers, syn]
    if columns is None:
        columns = [c for c in members.columns
                   if all(c in f.columns for f in frames) and np.issubdtype(members[c].dtype, np.number)]
        for c in (categorical or []):
            if c in members.columns and all(c in f.columns for f in frames) and c not in columns:
                columns.append(c)
    if not columns:
        raise ValueError("No shared columns to run the attack on.")
    categorical = [c for c in (categorical or []) if c in columns]
    continuous = [c for c in columns if c not in categorical]

    # Encode the three frames identically: continuous standardised on the members (so a
    # degenerate synthetic can't distort the scale), categoricals one-hot across all three
    # together (aligned dummy columns, left 0/1 -> no false ordinality).
    mem_blocks, non_blocks, syn_blocks = [], [], []
    if continuous:
        scaler = StandardScaler().fit(members[continuous].to_numpy())
        mem_blocks.append(scaler.transform(members[continuous].to_numpy()))
        non_blocks.append(scaler.transform(nonmembers[continuous].to_numpy()))
        syn_blocks.append(scaler.transform(syn[continuous].to_numpy()))
    if categorical:
        n_mem, n_non = len(members), len(nonmembers)
        dummies = pd.get_dummies(
            pd.concat([members[categorical], nonmembers[categorical], syn[categorical]], ignore_index=True),
            columns=categorical,
        )
        mem_blocks.append(dummies.iloc[:n_mem].to_numpy(dtype=float))
        non_blocks.append(dummies.iloc[n_mem:n_mem + n_non].to_numpy(dtype=float))
        syn_blocks.append(dummies.iloc[n_mem + n_non:].to_numpy(dtype=float))
    mem_mat = np.hstack(mem_blocks)
    non_mat = np.hstack(non_blocks)
    syn_mat = np.hstack(syn_blocks)

    # GAN-Leaks score: proximity to the nearest synthetic record (query real -> synthetic).
    nn = NearestNeighbors(n_neighbors=1).fit(syn_mat)
    d_mem = nn.kneighbors(mem_mat)[0][:, 0]
    d_non = nn.kneighbors(non_mat)[0][:, 0]

    def _scores_labels(dm, dn):
        # closer to synthetic -> more member-like, so negate distance
        return np.concatenate([-dm, -dn]), np.concatenate([np.ones(len(dm)), np.zeros(len(dn))])

    def _auc(dm, dn):
        scores, labels = _scores_labels(dm, dn)
        return float(roc_auc_score(labels, scores))

    def _tpr_at_fpr(dm, dn, target_fpr):
        # TPR at a fixed FPR: interpolate the ROC curve. Tail/worst-case leakage metric --
        # unlike AUC it does not average over easy records, so it catches a few memorised rows.
        scores, labels = _scores_labels(dm, dn)
        fpr, tpr, _ = roc_curve(labels, scores)
        return float(np.interp(target_fpr, fpr, tpr))

    targets = None if tpr_at_fpr is None else np.atleast_1d(tpr_at_fpr).astype(float)

    out = {
        "mia_auc": _auc(d_mem, d_non),
        "n_members": int(len(d_mem)),
        "n_nonmembers": int(len(d_non)),
    }
    if targets is not None:
        for t in targets:
            out[f"tpr@fpr={t:g}"] = _tpr_at_fpr(d_mem, d_non, t)

    if bootstrap:
        # Resample member and non-member distances (with replacement, separately) and re-score.
        # The synthetic reference is fixed, so the CI reflects the finite real sets -- the right
        # notion, since resampling *who is a member* would require retraining the generator.
        # AUC and every TPR@FPR share the same resamples so their CIs are mutually consistent.
        rng = np.random.default_rng(random_state)
        boot_auc = []
        boot_tpr = {} if targets is None else {t: [] for t in targets}
        for _ in range(n_bootstrap):
            bm = rng.choice(d_mem, size=len(d_mem), replace=True)
            bn = rng.choice(d_non, size=len(d_non), replace=True)
            boot_auc.append(_auc(bm, bn))
            for t in boot_tpr:
                boot_tpr[t].append(_tpr_at_fpr(bm, bn, t))
        out["ci_low"], out["ci_high"] = (float(v) for v in np.percentile(boot_auc, [2.5, 97.5]))
        if return_boot_auc:
            out["boot_auc"] = np.asarray(boot_auc, dtype=float)
        for t, vals in boot_tpr.items():
            lo, hi = np.percentile(vals, [2.5, 97.5])
            out[f"tpr@fpr={t:g}_ci_low"], out[f"tpr@fpr={t:g}_ci_high"] = float(lo), float(hi)

    return out


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