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
import surv_hivae, surv_gan, surv_vae

import os
import json
import datetime
import uuid

from synthcity.utils.constants import DEVICE
print('Device :', DEVICE)


def run(dataset_name, generator_name):

    current_path = os.getcwd()  # Get current working directory
    parent_path = os.path.dirname(current_path)

    data_file_control= parent_path + "/dataset/" + dataset_name + "/data_control.csv"
    feat_types_file_control = parent_path + "/dataset/" + dataset_name + "/data_types_control.csv"
    data_file_treated= parent_path + "/dataset/" + dataset_name + "/data_treated.csv"
    feat_types_file_treated= parent_path + "/dataset/" + dataset_name + "/data_types_treated.csv"

    # If the dataset has no missing data, leave the "miss_file" variable empty
    miss_file = parent_path + "dataset/" + dataset_name + "/Missing.csv"
    true_miss_file = None

    # Load and transform control data
    df_init_control_encoded, feat_types_dict, miss_mask_control, true_miss_mask_control, _ = data_processing.read_data(data_file_control,
                                                                                                                feat_types_file_control,
                                                                                                                miss_file, true_miss_file)
    data_init_control_encoded = torch.from_numpy(df_init_control_encoded.values)
    data_init_control = data_processing.discrete_variables_transformation(data_init_control_encoded, feat_types_dict)

    # Load and transform treated data
    df_init_treated_encoded, _, _, _, _ = data_processing.read_data(data_file_treated, feat_types_file_treated, miss_file, true_miss_file)
    data_init_treated_encoded = torch.from_numpy(df_init_treated_encoded.values)
    data_init_treated = data_processing.discrete_variables_transformation(data_init_treated_encoded, feat_types_dict)

    fnames = ['time', 'censor'] + pd.read_csv(feat_types_file_control)["name"].to_list()[1:]

    # Format data in dataframe
    df_init_treated = pd.DataFrame(data_init_treated.numpy(), columns=fnames)
    df_init_control = pd.DataFrame(data_init_control.numpy(), columns=fnames)

    # Update the data
    df_init_treated["treatment"] = 1
    df_init_control["treatment"] = 0
    df_init = pd.concat([df_init_control, df_init_treated], ignore_index=True)

    # Parameters of the optuna study
    metric_optuna = "survival_km_distance" # metric to optimize in optuna
    method_hyperopt = "train_full_gen_full"
    n_splits = 5 # number of splits for cross-validation
    n_generated_dataset = 200 # number of generated datasets per fold to compute the metric
    name_config = dataset_name

    generators_dict = {"HI-VAE_weibull" : surv_hivae,
                    "HI-VAE_piecewise" : surv_hivae,
                    "HI-VAE_lognormal" : surv_hivae,
                    "Surv-GAN" : surv_gan,
                    "Surv-VAE" : surv_vae, 
                    "HI-VAE_weibull_prior" : surv_hivae, 
                    "HI-VAE_piecewise_prior" : surv_hivae}
    
    # Set a unique working directory for this job
    original_dir, work_dir = setup_unique_working_dir("parallel_runs")
    os.chdir(work_dir)  # Switch to private work dir
    print("Working directory:", work_dir)
    print("Original directory:", original_dir)

    # Create directories for optuna results
    if not os.path.exists(parent_path + "/dataset/" + dataset_name + "/optuna_results"):
        os.makedirs(parent_path + "/dataset/" + dataset_name + "/optuna_results")

    best_params_dict, study_dict = {}, {}
    n_trials = 150
    epochs = 10000
    print("{} trials for {}...".format(n_trials, generator_name))
    study_name = parent_path + "/dataset/" + dataset_name + "/optuna_results/optuna_study_traincontrol_{}_ntrials{}_{}_{}".format(name_config, n_trials, metric_optuna, generator_name)
    best_params_file = parent_path + "/dataset/" + dataset_name + "/optuna_results/best_params_traincontrol_{}_ntrials{}_{}_{}.json".format(name_config, n_trials, metric_optuna, generator_name)
    db_file = study_name + ".db"
    if os.path.exists(db_file):
        print("This optuna study ({}) already exists for {}. We will use this existing file.".format(db_file, generator_name))
    else: 
        print("Creating new optuna study for {}...".format(generator_name))

    if generator_name in ["HI-VAE_lognormal", "HI-VAE_weibull", "HI-VAE_piecewise", "HI-VAE_weibull_prior", "HI-VAE_piecewise_prior"]:
        feat_types_dict_ext = feat_types_dict.copy()
        for i in range(len(feat_types_dict)):
            if feat_types_dict_ext[i]['name'] == "survcens":
                if generator_name in ["HI-VAE_weibull", "HI-VAE_weibull_prior"]:
                    feat_types_dict_ext[i]["type"] = 'surv_weibull'
                elif generator_name in ["HI-VAE_lognormal"]:
                    feat_types_dict_ext[i]["type"] = 'surv'
                else:
                    feat_types_dict_ext[i]["type"] = 'surv_piecewise'
        if generator_name in ["HI-VAE_weibull_prior", "HI-VAE_piecewise_prior"]:
            gen_from_prior = True
        else:
            gen_from_prior = False
        best_params, study = generators_dict[generator_name].optuna_hyperparameter_search(df_init_control_encoded,
                                                                                        miss_mask_control, 
                                                                                        true_miss_mask_control,
                                                                                        feat_types_dict_ext, 
                                                                                        n_generated_dataset, 
                                                                                        n_splits=n_splits,
                                                                                        n_trials=n_trials, 
                                                                                        columns=fnames,
                                                                                        generator_name=generator_name,
                                                                                        epochs=epochs,
                                                                                        metric=metric_optuna,
                                                                                        study_name=study_name, 
                                                                                        method=method_hyperopt, 
                                                                                        gen_from_prior=gen_from_prior)
        best_params_dict[generator_name] = best_params
        study_dict[generator_name] = study
        with open(best_params_file, "w") as f:
            json.dump(best_params, f)
    else: 
        best_params, study = generators_dict[generator_name].optuna_hyperparameter_search(data_init_control, 
                                                                                        columns=fnames, 
                                                                                        target_column="censor", 
                                                                                        time_to_event_column="time", 
                                                                                        n_generated_dataset=n_generated_dataset, 
                                                                                        n_splits=n_splits,
                                                                                        n_trials=n_trials,
                                                                                        metric=metric_optuna,
                                                                                        study_name=study_name, 
                                                                                        method=method_hyperopt)
        best_params_dict[generator_name] = best_params
        study_dict[generator_name] = study
        with open(best_params_file, "w") as f:
            json.dump(best_params, f)



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
    dataset_name = "Aids"
    generators_sel = ["HI-VAE_weibull", "HI-VAE_piecewise", "Surv-GAN", "Surv-VAE", "HI-VAE_weibull_prior", "HI-VAE_piecewise_prior"]
    gen = generators_sel[int(sys.argv[1])]
    run(dataset_name , gen)