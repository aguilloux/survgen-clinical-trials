#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Converted to PyTorch

Created on Mon Feb 17 20:35:11 2025

@author: Van Tuan NGUYEN
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
sns.set(style="whitegrid", font="STIXGeneral", context="talk", palette="colorblind")

# diverging colormap going orange -> light -> blue (low = orange, high = blue)
BLUE_ORANGE = LinearSegmentedColormap.from_list("blue_orange",
                                                ["#d8761d", "#f7f7f7", "#2166ac"])

from sksurv.nonparametric import kaplan_meier_estimator
from lifelines import KaplanMeierFitter



def plot_data_compare(data_list, feat_types_dict, feat_comparison_name="group"):
    """
    Visualize features across multiple groups using violin/count plots.
    
    Args:
        data_list (list of pd.DataFrame): List of dataframes to compare.
        feat_types_dict (list of dict): Feature descriptions with keys: 'name', 'type', 'dim'.
        feat_comparison_name (str): Column name indicating group (automatically added).
    """
    # Concaténer tous les DataFrames et ajouter l'information de groupe
    # combined_data = []
    # for i, df in enumerate(data_list):
    #     df_copy = df.copy()
    #     df_copy[feat_comparison_name] = f"group_{i}"
    #     combined_data.append(df_copy)
    
    full_data = pd.concat(data_list, ignore_index=True)

    num_features = len(feat_types_dict) 
    n_cols = num_features // 2 + num_features % 2
    _, axes = plt.subplots(n_cols, 2, figsize=(18, 2.5 * num_features))
    axes = axes.flatten()
    
    for i, feature in enumerate(feat_types_dict):
        ax = axes[i]
        feat_name = feature['name']
        feature_type = feature['type']
        
        if feature_type in ['cat', 'ordinal']:
            # full_data[feat_comparison_name] = full_data[feat_comparison_name].astype("category")
            # if feat_types_dict[i]["nclass"] == 2:
            #     full_data[feat_name] = pd.Categorical(full_data[feat_name], categories=[0, 1])
            sns.countplot(data=full_data, x=feat_name, hue=feat_comparison_name, alpha=0.8, ax=ax, stat="percent")
            ax.set_title(f"Count plot of {feat_name} ({feature_type})", fontsize=14, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("Count")

        
        elif feature_type.startswith("surv"):
            print(f"Skipping survival-type feature '{feat_name}' for plotting.")
            ax.set_visible(False)

        else:
            sns.violinplot(data=full_data, x=feat_comparison_name, y=feat_name, ax=ax)
            ax.set_title(f"Distribution of {feat_name} ({feature_type})", fontsize=14, fontweight="bold")
            ax.set_xlabel(feat_comparison_name)
            ax.set_ylabel(feat_name)

        ax.grid(True)

    # Supprimer axes inutilisés s’il y en a
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_data(data, feat_types_dict,feat_comparison_name=None):
    """
    Visualize different data types.

    Args:
    - data (np.ndarray): Input data (shape: [n_samples, n_features]).
    - feat_types_dict (list): List of feature type dictionaries (e.g., [{'type': 'real', 'dim': 1}, ...]).
    - feat_comparison_name (str): 
    """
    num_features = len(feat_types_dict)
    if feat_comparison_name is not None:
        n_cols = (num_features - 1) // 2 + (num_features - 1) % 2
    else:
        n_cols = num_features // 2 + num_features % 2
    _, axes = plt.subplots(n_cols, 2, figsize=(18, 2.5 * num_features))
    plt.subplots_adjust(wspace=0.2, hspace=0.35)
    
    feat_idx = 0
    for i, feature in enumerate(feat_types_dict):
        if feature['name'] == feat_comparison_name:
            feat_comparison_index = feat_idx
        feat_idx += int(feature['dim'])

    feat_idx = 0
    for i, feature in enumerate(feat_types_dict):
        if feat_comparison_name is not None:
            if feat_idx == feat_comparison_index:
                feat_idx +=1
                continue
            else:
                if feat_idx < feat_comparison_index:
                    ax= axes[i // 2, i % 2]
                else:
                    ax= axes[(i-1) // 2, (i-1) % 2]
        else:
            ax= axes[i // 2, i % 2]

        feature_type = feature['type']
        feat_name = feature['name']
        
        if feature_type in ['cat', 'ordinal']:  # Count, ordinal & categorical data
            feature_data = pd.DataFrame(data[:, [feat_idx, feat_comparison_index]], columns=[feat_name, feat_comparison_name])
            sns.countplot(data=feature_data, x=feat_name, hue=feat_comparison_name, alpha=0.8, legend=True, ax=ax)
            ax.set_title(f"Count plot of {feat_name} ({feature_type})", fontsize=16, fontweight="bold")
            ax.legend(title=feat_comparison_name).set_visible(True)
            
            n_class = np.unique(feature_data.values).shape[0]
            if n_class > 20:
                # Dynamically reduce the number of x-ticks
                ticks = ax.get_xticks()  # Get original tick positions
                labels = ax.get_xticklabels()  # Get original labels

                step = max(1, len(labels) // 10)  # Show every 10th label
                reduced_ticks = ticks[::step]
                reduced_labels = labels[::step]

                ax.set_xticks(reduced_ticks)  # Ensure same number of locations
                ax.set_xticklabels([label.get_text() for label in reduced_labels])  # Set labels
            ax.set_xlabel("")

        elif feature_type in ["surv", 'surv_weibull', 'surv_loglog', 'surv_piecewise']:
            
            survival_time, censoring_indicator, treat  =  data[:, list(range((feat_idx), (feat_idx) + 2)) + [feat_comparison_index]].T
    
            time_S1, survival_prob_S1, conf_int = kaplan_meier_estimator((censoring_indicator[treat==1]==1), survival_time[treat==1], conf_type="log-log")
            time_C1, survival_prob_C1, conf_int = kaplan_meier_estimator((1-censoring_indicator[treat==1]==1), survival_time[treat==1], conf_type="log-log")
            time_S0, survival_prob_S0, conf_int = kaplan_meier_estimator((censoring_indicator[treat==0]==1), survival_time[treat==0], conf_type="log-log")
            time_C0, survival_prob_C0, conf_int = kaplan_meier_estimator((1-censoring_indicator[treat==0]==1), survival_time[treat==0], conf_type="log-log")
            

            label_cens_0 = "Cens. time(" + feat_comparison_name + "=0)"
            label_cens_1 = "Cens. time(" + feat_comparison_name + "=1)"
            label_time_0 = "Surv. time(" + feat_comparison_name + "=0)"
            label_time_1 = "Surv. time(" + feat_comparison_name + "=1)"

            ax.step(time_S1, survival_prob_S1, where="post", label=label_time_1, c='r')
            ax.step(time_C1, survival_prob_C1, where="post", label=label_cens_1, c='b')
            ax.step(time_S0, survival_prob_S0, where="post", label=label_time_0, c='m')
            ax.step(time_C0, survival_prob_C0, where="post", label=label_cens_0, c='c')
            ax.legend().set_visible(True)
            ax.set_xlabel("")

            feat_idx += 1

        else:
            feature_data = pd.DataFrame(data[:, [feat_idx,feat_comparison_index]], columns=[feat_name,feat_comparison_name])
            feature_data[feat_comparison_name] = feature_data[feat_comparison_name].astype('category')
            sns.violinplot(feature_data,  x=feat_comparison_name, y=feat_name, ax=ax)
            
            ax.set_title(f"Distribution plot of {feat_name} ({feature_type})", fontsize=16, fontweight="bold")
            ax.set_xlabel(feat_comparison_name, fontsize=16, fontweight="semibold")
        
        feat_idx += 1
        # Enhance visualization
        ax.grid(True)
        ax.set_ylabel("Count", fontsize=16, fontweight="semibold")

    plt.show()

def plot_loss_evolution(loss_track, title, xlabel, ylabel):
    """
    Plot the loss curve

    Parameters
    ----------
    loss_track :  `np.ndarray`, shape=(n_samples, 2)
        Normal array of survival labels

    title : `str`
        Title of the figure

    xlabel : `str`
        Label of x axis

    ylabel : `str`
        Label of y axis
    """
    # plt.figure(figsize=(8, 4))
    _, ax = plt.subplots(1, 1, figsize=(10, 5))
    sns.lineplot(loss_track, ax=ax)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()


# Function to print loss metrics
def print_loss(epoch, start_time, ELBO, avg_KL_s, avg_KL_z):
    print("Epoch: [%2d]  time: %4.4f, ELBO_train: %.8f, KL_z: %.8f, KL_s: %.8f, reconstruction loss: %.8f"
          % (epoch, time.time() - start_time, ELBO, avg_KL_z, avg_KL_s, ELBO + avg_KL_z + avg_KL_s))


def visualize_general_perf(scores, metrics, title = None):
    """
    Generate boxplots to visualize performance scores across different generators.

    Args:
        scores (DataFrame): Performance metrics for different synthetic data generators.
        metrics (list of str): List of column names (metrics) to plot.
    """
    num_metrics = len(metrics)
    n_learners = len(np.unique(scores['generator'].values))
    fig, axs = plt.subplots(1, num_metrics, figsize=(max(3 * num_metrics * n_learners, 6), 6))

    if num_metrics == 1:
        axs = [axs]  # ensure axs is iterable

    for i, ax in enumerate(axs):
        # Format axis spines
        metric_name, opt = metrics[i]
        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_edgecolor('black')

        sns.boxplot(data=scores, x='generator', y=metric_name, ax=ax,
                    linewidth = 2, saturation = 1, palette = 'colorblind', 
                    width = 1, gap = 0.15, whis = 1.5, linecolor="Black")
        ax.set_xlabel('')
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)

        opt_arrow = ""
        if opt == "max":
            opt_arrow = ' \u2192'
        elif opt == "min":
            opt_arrow = ' \u2190'

        ax.set_ylabel(metric_name + opt_arrow, fontweight="semibold", fontsize=20)
    if title is not None:
        plt.suptitle(title, y=0.8, fontsize=20, fontweight="semibold")
    plt.tight_layout(pad=3)
    plt.show()

def _whisker_extent(values, whis=1.5):
    """
    Tukey whisker range for a 1-D array, matching seaborn/matplotlib boxplots.

    Returns (low, high): the most extreme data points within ``whis`` * IQR of
    the box (Q1/Q3). NaNs are dropped; returns (None, None) if nothing is left.
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return None, None
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    low = values[values >= q1 - whis * iqr].min()
    high = values[values <= q3 + whis * iqr].max()
    return low, high


def visualize_grouped_perf(scores, dict_metrics, fontsize=20, panel_width=None,
                           height=6, suptitle=None):
    """
    Combine several metric groups (e.g. Fidelity / Utility / Privacy) side by side in
    a single figure. Every individual metric panel is given the same width regardless
    of how many metrics each group holds, and all text (group labels, axis labels and
    tick labels) is rendered at the same size. Each group name is left-aligned above
    its leftmost panel (rather than centered over the group) so it is always clear
    which metrics belong to which category, even when groups hold an unequal number
    of metrics.

    Args:
        scores (DataFrame): performance metrics for the different generators.
        dict_metrics (dict): {group_title: [[metric_name, opt], ...], ...}, where
            ``opt`` controls the y-label annotation. Use "min" / "max" for a
            plain direction arrow (← / →). Alternatively pass a float to mark an
            ideal target value (e.g. 0.5 for identifiability_score): the arrow
            then points toward the target when every generator's box sits on one
            side of it, and is replaced by a dotted red reference line at the
            target when the pooled whisker range brackets it.
        fontsize (int): font size used for every text element.
        panel_width (float): width in inches allotted to each metric panel. If None,
            it scales with the number of generators so boxes are never squished.
        height (float): figure height in inches.
        suptitle (str): optional overall figure title.
    """
    group_names = list(dict_metrics.keys())
    counts = [len(dict_metrics[g]) for g in group_names]
    total_metrics = sum(counts)
    n_learners = len(np.unique(scores['generator'].values))

    if panel_width is None:
        panel_width = max(3 * n_learners, 3)

    fig = plt.figure(figsize=(panel_width * total_metrics, height), layout="constrained")
    # one sub-figure per group, side by side; width_ratios = metric count per group
    # so every panel ends up the same width
    subfigs = fig.subfigures(1, len(group_names), width_ratios=counts, wspace=0.02)
    if len(group_names) == 1:
        subfigs = [subfigs]

    for subfig, group_name in zip(subfigs, group_names):
        metrics = dict_metrics[group_name]
        # squeeze=False keeps axs 2-D even for a single-metric group
        axs = subfig.subplots(1, len(metrics), squeeze=False)[0]

        for ax, (metric_name, opt) in zip(axs, metrics):
            for spine in ax.spines.values():
                spine.set_linewidth(2)
                spine.set_edgecolor('black')

            sns.boxplot(data=scores, x='generator', y=metric_name, ax=ax,
                        linewidth=2, saturation=1, palette='colorblind',
                        width=1, gap=0.15, whis=1.5, linecolor="Black")

            if isinstance(opt, (int, float)) and not isinstance(opt, bool):
                # metric with an ideal target value rather than a min/max goal.
                # Pool the whisker extents of every generator's box, then compare
                # the overall [low, high] range against the target: if it brackets
                # the target there is no meaningful direction (mark it with a
                # dotted red reference line instead); otherwise the arrow points
                # the way that moves the boxes toward the target.
                ideal = float(opt)
                lows, highs = [], []
                for _, grp in scores.groupby('generator')[metric_name]:
                    lo, hi = _whisker_extent(grp.values, whis=1.5)
                    if lo is not None:
                        lows.append(lo)
                        highs.append(hi)
                overall_low = min(lows) if lows else ideal
                overall_high = max(highs) if highs else ideal
                if overall_low <= ideal <= overall_high:
                    opt_arrow = ""
                    ax.axhline(ideal, ls=":", color="red", lw=2, zorder=10)
                elif overall_high < ideal:
                    opt_arrow = ' →'  # boxes below target -> increase toward it
                else:
                    opt_arrow = ' ←'  # boxes above target -> decrease toward it
            else:
                opt_arrow = {"max": ' →', "min": ' ←'}.get(opt, "")
            ax.set_xlabel('')
            ax.set_ylabel(metric_name + opt_arrow, fontweight="semibold", fontsize=fontsize)
            ax.tick_params(axis='x', labelsize=fontsize)
            ax.tick_params(axis='y', labelsize=fontsize)

        # left-align the group label (instead of centering it) so it sits above
        # the group's leftmost panel; reads as "everything to the right of this
        # label, up to the next group, belongs to this category" and keeps e.g.
        # K-map score visibly under Privacy rather than appearing under Utility
        subfig.suptitle(group_name, x=0.0, ha='left',
                        fontsize=fontsize, fontweight="bold")

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=fontsize, fontweight="bold")
    plt.show()


def visualize_replicability_perf(scores):
    """
    Generate boxplots to visualize performance scores across different generators.

    Args:
        scores (DataFrame): Performance metrics for different synthetic data generators.
    """
    metric_names = scores.columns.values[2:]
    num_metrics = len(metric_names)
    fig, axs = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 6))

    if num_metrics == 1:
        axs = [axs]  # ensure axs is iterable

    for i, ax in enumerate(axs):
        # Format axis spines
        metric_name = metric_names[i]
        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_edgecolor('black')

        sns.lineplot(data=scores, x='Nb generated datasets', y=metric_name,
                     hue="Generator", ax=ax, palette = 'colorblind')
        ax.set_xlabel('Nb generated datasets', fontsize=20, fontweight="semibold")
        ax.set_ylabel(metric_name, fontsize=20, fontweight="semibold")
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        ax.set_ylim(0, 1.05)
    plt.tight_layout(pad=3)
    plt.show()


def plot_variable_heatmap(dict_list, names, title=None, cmap=BLUE_ORANGE,
                          vmin=0.0, vmax=1.0, figsize=None):
    """
    Compare several methods on a per-variable score with an annotated heatmap.

    Each method is summarised by a dict mapping variable name -> score in [0, 1]
    (typically the output of ``metrics.variable_pct_failed_tests``: the fraction
    of generated datasets for which the per-variable TableOne test found no
    significant difference from the real data). Methods become the rows and the
    variables (the dict keys, assumed identical across methods) become the
    columns. Every tile shows its value and is coloured on a fixed 0-1 scale, so
    colours stay comparable across figures.

    The variable names and the "Variable" axis label are placed along the top
    edge, while the (optional) title sits underneath the heatmap.

    Args:
        dict_list (list of dict): one ``{variable: score}`` dict per method (e.g.
            ``[a_test, b_test]``). All dicts are expected to share the same keys.
        names (list of str): method names, one per dict, used as the row labels.
            Must be the same length as ``dict_list``.
        title (str): optional title, drawn at the bottom of the figure.
        cmap: seaborn/matplotlib colormap. Default ``BLUE_ORANGE`` (low = orange,
            high = blue).
        vmin, vmax (float): fixed colour-scale bounds. Default 0-1.
        figsize (tuple): optional (width, height) in inches; scales with the
            number of variables/methods when left as None.
    """
    if len(dict_list) != len(names):
        raise ValueError(f"dict_list and names must have the same length "
                         f"({len(dict_list)} != {len(names)}).")

    # rows = methods (names), columns = variables (dict keys, insertion order kept)
    table = pd.DataFrame(dict_list, index=names)

    if figsize is None:
        figsize = (max(1.2 * table.shape[1] + 2, 6), max(0.8 * table.shape[0] + 1, 3))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(table, annot=True, fmt=".2f", cmap=cmap, vmin=vmin, vmax=vmax,
                linewidths=0.5, linecolor="white", cbar=False, ax=ax)

    # variable names (tick labels) and the "Variable" axis label both on top
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Variable", fontweight="semibold")
    ax.set_ylabel("Method", fontweight="semibold")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)

    # title at the very bottom of the figure; reserve space so it is never clipped
    plt.tight_layout(rect=[0, 0.12, 1, 1] if title is not None else None)
    if title is not None:
        fig.text(0.5, 0.02, title, ha="center", va="bottom", fontweight="bold")

    plt.show()