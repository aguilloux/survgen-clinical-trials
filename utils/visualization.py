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
import seaborn as sns
sns.set(style="whitegrid", font="STIXGeneral", context="talk", palette="colorblind")

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
    fig, axs = plt.subplots(1, num_metrics, figsize=(3 * num_metrics * n_learners, 6))

    if num_metrics == 1:
        axs = [axs]  # ensure axs is iterable

    for i, ax in enumerate(axs):
        # Format axis spines
        metric_name, opt = metrics[i]
        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_edgecolor('black')

        sns.boxplot(data=scores, x='generator', y=metric_name, ax=ax,
                    linewidth = 3, saturation = 1, palette = 'colorblind', 
                    width = 1, gap = 0.15, whis = 0.8, linecolor="Black")
        ax.set_xlabel('')
        ax.set_ylabel(metric_name, fontsize=20, fontweight="semibold")
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        if opt == "max":
            ax.legend(title='Maximize \u2191', title_fontsize=15)
        else:
            ax.legend(title='Minimize \u2193', title_fontsize=15)
    if title is not None:
        plt.suptitle(title, y=0.8, fontsize=20, fontweight="semibold")
    plt.tight_layout(pad=3)
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