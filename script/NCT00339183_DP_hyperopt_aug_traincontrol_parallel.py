import numpy as np
import pandas as pd
import torch

import sys
from pathlib import Path
module_path = Path.cwd().parent / 'utils'
sys.path.append(str(module_path))
import data_processing
from simulations import *
module_path = Path.cwd().parent / 'execute'
sys.path.append(str(module_path))
import surv_hivae

import os
import json
import sys
import datetime
import uuid

from synthcity.utils.constants import DEVICE
print('Device :', DEVICE)


def run(generator_name, target_epsilon):

    list_n_samples_control = [1.0] # [(1/3), (2/3), 1.0]

    current_path = os.getcwd()  # Get current working directory
    parent_path = os.path.dirname(current_path)
    if not os.path.exists(parent_path + "/dataset"):
        os.makedirs(parent_path + "/dataset/")

    # Save the data
    dataset_name = "NCT00339183"
    if not os.path.exists(parent_path + "/dataset/" + dataset_name):
        os.makedirs(parent_path + "/dataset/" + dataset_name)

    # Set a unique working directory for this job
    original_dir, work_dir = setup_unique_working_dir("parallel_runs")
    print("Working directory:", work_dir)
    print("Original directory:", original_dir)

    for d, perc_control in enumerate(list_n_samples_control):

        data_file_control = parent_path + "/dataset/" + dataset_name + "/data_control.csv"
        feat_types_file_control = parent_path + "/dataset/" + dataset_name + "/data_types_control.csv"
        data_file_treated = parent_path + "/dataset/" + dataset_name + "/data_treated.csv"
        feat_types_file_treated = parent_path + "/dataset/" + dataset_name + "/data_types_treated.csv"

        # If the dataset has no missing data, leave the "miss_file" variable empty
        miss_file = parent_path + "dataset/" + dataset_name + "/Missing.csv"
        true_miss_file = None

        # Load and transform control data
        df_init_control_encoded, feat_types_dict, miss_mask_control, true_miss_mask_control, _ = data_processing.read_data(data_file_control,
                                                                                                                        feat_types_file_control,
                                                                                                                        miss_file, true_miss_file)

        n_generated_samples_control = df_init_control_encoded.shape[0]
        n_samples_control = int(perc_control * n_generated_samples_control)
        print("n_samples_control:", n_samples_control)
        df_init_control_encoded = df_init_control_encoded.iloc[:n_samples_control]
        data_init_control_encoded = torch.from_numpy(df_init_control_encoded.values)
        data_init_control = data_processing.discrete_variables_transformation(data_init_control_encoded, feat_types_dict)

        fnames = ['time', 'censor'] + pd.read_csv(feat_types_file_control)["name"].to_list()[1:]

        # Parameters of the optuna study
        method_HPO = "DetectXGB"
        optuna_version_name = method_HPO
        if method_HPO == "ValLoss":
            HPO_version = "validation_loss"  # "external_metrics" or "validation_loss"
        elif method_HPO == "DetectXGB":
            HPO_version = "external_metrics"
            metric_optuna = ["detection_xgb"]
        elif method_HPO == "SurvDist":
            HPO_version = "external_metrics"
            metric_optuna = ["survival_km_distance"]
        elif method_HPO == "Kmap_SurvDist":
            HPO_version = "external_metrics"
            metric_optuna = ["survival_km_distance", "k-map"]
        elif method_HPO == "IdfScore_SurvDist":
            HPO_version = "external_metrics"
            metric_optuna = ["survival_km_distance", "identifiability_score"]
        method_hyperopt = "train_full_gen_full"
        n_splits = 5  # number of splits for cross-validation
        n_generated_dataset = 200  # number of generated datasets per fold to compute the metric
        n_trials = 150
        # eps_tag = "DP_eps_{}".format(int(target_epsilon))
        name_config = "traincontrol_" + dataset_name + "_aug_Ncontrol{}%3".format((d + 1))

        generators_dict = {"HI-VAE_weibull": surv_hivae,
                        "HI-VAE_piecewise": surv_hivae}

        # Create directories for optuna results
        if not os.path.exists(parent_path + "/dataset/" + dataset_name + "/optuna_results"):
            os.makedirs(parent_path + "/dataset/" + dataset_name + "/optuna_results")

        # Load the non-DP best params and fix everything except lr and batch_size
        # base_generator_name = generator_name.replace("_DP", "")
        source_name_config = "best_params_{}_ntrials{}_{}_{}.json".format(name_config, n_trials, method_HPO, generator_name)
        source_best_params_file = None
        for f in os.listdir(parent_path + "/dataset/" + dataset_name + "/optuna_results"):
            if f.endswith(generator_name + ".json") and (source_name_config in f):
                source_best_params_file = f
        if source_best_params_file is None:
            raise FileNotFoundError("Could not find non-DP best params file matching '{}...{}.json'.".format(source_name_config, generator_name))
        with open(parent_path + "/dataset/" + dataset_name + "/optuna_results/" + source_best_params_file, "r") as f:
            source_best_params = json.load(f)
        dict_optimal_fix_params = {key: value for key, value in source_best_params.items() if key not in ['lr', 'batch_size']}

        best_params_dict, study_dict = {}, {}
        print("{} trials for {} (target_epsilon={})...".format(n_trials, generator_name, target_epsilon))
        study_name = parent_path + "/dataset/" + dataset_name + "/optuna_results/optuna_study_{}_ntrials{}_{}_{}_DP_eps{}".format(name_config, n_trials, optuna_version_name, generator_name, int(target_epsilon))
        best_params_file = parent_path + "/dataset/" + dataset_name + "/optuna_results/best_params_{}_ntrials{}_{}_{}_DP_eps{}.json".format(name_config, n_trials, optuna_version_name, generator_name, int(target_epsilon))
        db_file = study_name + ".db"
        if os.path.exists(db_file):
            print("This optuna study ({}) already exists for {}. We will use this existing file.".format(db_file, generator_name))
        else:
            print("Creating new optuna study for {}...".format(generator_name))

        os.chdir(work_dir)  # Switch to private work dir

        feat_types_dict_ext = feat_types_dict.copy()
        for i in range(len(feat_types_dict)):
            if feat_types_dict_ext[i]['name'] == "survcens":
                if "HI-VAE_weibull" in generator_name:
                    feat_types_dict_ext[i]["type"] = 'surv_weibull'
                else:
                    feat_types_dict_ext[i]["type"] = 'surv_piecewise'
        gen_from_prior = "_prior" in generator_name
        differential_privacy = True
        diffusion = "_diffusion" in generator_name

        best_params, study = generators_dict[generator_name].optuna_hyperparameter_search(df_init_control_encoded,
                                                                                        miss_mask_control,
                                                                                        true_miss_mask_control,
                                                                                        feat_types_dict_ext,
                                                                                        n_generated_dataset,
                                                                                        n_splits=n_splits,
                                                                                        n_trials=n_trials,
                                                                                        columns=fnames,
                                                                                        generator_name=generator_name,
                                                                                        metric=metric_optuna,
                                                                                        study_name=study_name,
                                                                                        method=method_hyperopt,
                                                                                        gen_from_prior=gen_from_prior,
                                                                                        seed=10,
                                                                                        target_epsilon=float(target_epsilon),  # None if not DP, otherwise the target epsilon for the DP generators
                                                                                        target_delta=1e-5,
                                                                                        tune_params=['epochs', 'lr', 'batch_size', 'max_grad_norm'],
                                                                                        fixed_params=dict_optimal_fix_params,  # these parameters will be fixed to the specified value and not tuned
                                                                                        norm_mode="global",
                                                                                        differential_privacy=differential_privacy,
                                                                                        diffusion=diffusion,
                                                                                        do_prune=False,
                                                                                        apply_rounding=True)
        best_params_full = dict_optimal_fix_params | best_params  # keep the non-DP fixed params alongside the tuned DP ones
        best_params_dict[generator_name] = best_params_full
        study_dict[generator_name] = study
        with open(best_params_file, "w") as f:
            json.dump(best_params_full, f)

        os.chdir(original_dir)


def setup_unique_working_dir(base_dir="experiments"):
    original_dir = os.getcwd()  # Save original dir
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    uid = uuid.uuid4().hex[:8]
    work_dir = os.path.join(base_dir, f"run_{timestamp}_{uid}")
    os.makedirs(work_dir, exist_ok=True)
    # os.chdir(work_dir)  # Switch to private work dir
    return original_dir, work_dir  # Return the original dir


if __name__ == "__main__":
    generators_sel = ["HI-VAE_weibull", "HI-VAE_piecewise"]
    target_epsilons = [1, 3, 5, 7, 10]
    jobs = [(generator_name, eps) for generator_name in generators_sel for eps in target_epsilons]
    job_id = int(sys.argv[1])
    generator_name, target_epsilon = jobs[job_id]
    run(generator_name, target_epsilon)
