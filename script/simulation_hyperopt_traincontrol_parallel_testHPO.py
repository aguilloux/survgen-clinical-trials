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
import sys
import datetime
import uuid

from synthcity.utils.constants import DEVICE
print('Device :', DEVICE)


def run(generator_name):

    n_samples = 600
    n_features_bytype = 6
    n_active_features = 3 
    treatment_effect = 0.
    p_treated = 0.5
    shape_T = 2.
    shape_C = 2.
    scale_C = 2.5
    scale_C_indep = 3.9
    feature_types_list = ["real", "cat"]
    independent = True
    data_types_create = True
    seed_data = 0

    control, treated, types = simulation(treatment_effect, n_samples, independent, feature_types_list,
                                         n_features_bytype, n_active_features, p_treated, shape_T,
                                         shape_C, scale_C, scale_C_indep, data_types_create, seed=seed_data)

    control = control.drop(columns='treatment')
    treated = treated.drop(columns='treatment')

    current_path = os.getcwd()  # Get current working directory
    parent_path = os.path.dirname(current_path)
    if not os.path.exists(parent_path + "/dataset"):
        os.makedirs(parent_path + "/dataset/")

    # Save the data
    dataset_name = "Simulations_indep_traincontrol"
    if not os.path.exists(parent_path + "/dataset/" + dataset_name):
        os.makedirs(parent_path + "/dataset/" + dataset_name)

    param_file = parent_path + "/dataset/" + dataset_name + "/params.txt"
    with open(param_file, "w") as f:
        f.write(f"n_samples = {n_samples}\n")
        f.write(f"n_features_bytype = {n_features_bytype}\n")
        f.write(f"n_active_features = {n_active_features}\n")
        f.write(f"treatment_effect = {treatment_effect}\n")
        f.write(f"p_treated = {p_treated}\n")
        f.write(f"shape_T = {shape_T}\n")
        f.write(f"shape_C = {shape_C}\n")
        f.write(f"scale_C = {scale_C}\n")
        f.write(f"scale_C_indep = {scale_C_indep}\n")
        f.write(f"feature_types_list = {feature_types_list}\n")
        f.write(f"independent = {independent}\n")
        f.write(f"data_types_create = {data_types_create}\n")
    
    data_file_control= parent_path + "/dataset/" + dataset_name + "/data_control.csv"
    feat_types_file_control = parent_path + "/dataset/" + dataset_name + "/data_types_control.csv"
    data_file_treated= parent_path + "/dataset/" + dataset_name + "/data_treated.csv"
    feat_types_file_treated= parent_path + "/dataset/" + dataset_name + "/data_types_treated.csv"

    # If the dataset has no missing data, leave the "miss_file" variable empty
    miss_file = parent_path + "dataset/" + dataset_name + "/Missing.csv"
    true_miss_file = None

    control.to_csv(data_file_control,index=False , header=False)
    types.to_csv(feat_types_file_control)
    treated.to_csv(data_file_treated,index=False , header=False)
    types.to_csv(feat_types_file_treated)

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

    fnames = types['name'][:-1].tolist()
    fnames.append("time")#.append("censor")
    fnames.append("censor")

    # Format data in dataframe
    df_init_treated = pd.DataFrame(data_init_treated.numpy(), columns=fnames)
    df_init_control = pd.DataFrame(data_init_control.numpy(), columns=fnames)

    # Update the data
    df_init_treated["treatment"] = 1
    df_init_control["treatment"] = 0
    df_init = pd.concat([df_init_control, df_init_treated], ignore_index=True)

    # Parameters of the optuna study
    HPO_version = "validation_loss" # "external_metrics" or "validation_loss"
    metric_optuna = ["survival_km_distance", "k-map"] # metric to optimize in optuna
    method_hyperopt = "train_full_gen_full"
    n_splits = 3 # number of splits for cross-validation
    n_generated_dataset = 200 # number of generated datasets per fold to compute the metric
    name_config = "simu_N{}_nfeat{}_t{}".format(n_samples, n_features_bytype, int(treatment_effect))

    seed_HPO_list = [10,11,12]
    for seed_HPO in seed_HPO_list:
        optuna_version_name = "ExMetrics2_seedData{}_seedHPO{}".format(seed_data, seed_HPO)
        # optuna_version_name = "ValLoss_seedData{}_seedHPO{}".format(seed_data, seed_HPO)

        generators_dict = {"HI-VAE_weibull" : surv_hivae,
                        "HI-VAE_piecewise" : surv_hivae,
                        "HI-VAE_lognormal" : surv_hivae,
                        "Surv-GAN" : surv_gan,
                        "Surv-VAE" : surv_vae, 
                        "HI-VAE_weibull_prior" : surv_hivae, 
                        "HI-VAE_piecewise_prior" : surv_hivae,
                        "HI-VAE_weibull_DP" : surv_hivae, 
                        "HI-VAE_piecewise_DP" : surv_hivae, 
                        "HI-VAE_weibull_diffuse": surv_hivae, 
                        "HI-VAE_piecewise_diffuse": surv_hivae,
                        "HI-VAE_weibull_diffuse_DP": surv_hivae, 
                        "HI-VAE_piecewise_diffuse_DP": surv_hivae}
        
        # Set a unique working directory for this job
        original_dir, work_dir = setup_unique_working_dir("parallel_runs")
        os.chdir(work_dir)  # Switch to private work dir
        print("Working directory:", work_dir)
        print("Original directory:", original_dir)

        # Create directories for optuna results
        if not os.path.exists(parent_path + "/dataset/" + dataset_name + "/optuna_results"):
            os.makedirs(parent_path + "/dataset/" + dataset_name + "/optuna_results")

        best_params_dict, study_dict = {}, {}
        # for generator_name in generators_sel:
        # n_trials = min(100, int(multiplier_trial * generators_dict[generator_name].get_n_hyperparameters(generator_name)))
        n_trials = 150
        print("{} trials for {}...".format(n_trials, generator_name))
        study_name = parent_path + "/dataset/" + dataset_name + "/optuna_results/optuna_study_{}_ntrials{}_{}_{}".format(name_config, n_trials, optuna_version_name, generator_name)
        best_params_file = parent_path + "/dataset/" + dataset_name + "/optuna_results/best_params_{}_ntrials{}_{}_{}.json".format(name_config, n_trials, optuna_version_name, generator_name)
        db_file = study_name + ".db"
        if os.path.exists(db_file):
            print("This optuna study ({}) already exists for {}. We will use this existing file.".format(db_file, generator_name))
        else: 
            print("Creating new optuna study for {}...".format(generator_name))

        if generator_name in ["HI-VAE_lognormal", "HI-VAE_weibull", "HI-VAE_piecewise", 
                            "HI-VAE_weibull_prior", "HI-VAE_piecewise_prior", 
                            "HI-VAE_weibull_DP", "HI-VAE_piecewise_DP",
                            "HI-VAE_weibull_diffuse", "HI-VAE_piecewise_diffuse", 
                            "HI-VAE_weibull_diffuse_DP", "HI-VAE_piecewise_diffuse_DP"]:
            feat_types_dict_ext = feat_types_dict.copy()
            for i in range(len(feat_types_dict)):
                if feat_types_dict_ext[i]['name'] == "survcens":
                    if "HI-VAE_weibull" in generator_name: # in ["HI-VAE_weibull", "HI-VAE_weibull_prior"]:
                        feat_types_dict_ext[i]["type"] = 'surv_weibull'
                    elif generator_name in ["HI-VAE_lognormal"]:
                        feat_types_dict_ext[i]["type"] = 'surv'
                    else:
                        feat_types_dict_ext[i]["type"] = 'surv_piecewise'
            gen_from_prior = "_prior" in generator_name
            differential_privacy = "_DP" in generator_name
            diffuse = "_diffuse" in generator_name
            if HPO_version == "external_metrics":
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
                                                                                                target_epsilon=1.0, # None if not DP, otherwise the target epsilon for the DP generators
                                                                                                target_delta=1e-5,
                                                                                                tune_params=None, # if None, the function will use the default hyperparameters to tune,
                                                                                                fixed_params={"epochs": 10000, "n_samples_gen": n_samples}, # these parameters will be fixed to the specified value and not tuned,
                                                                                                norm_mode="global",
                                                                                                screening_epochs=800,
                                                                                                n_startup_trials=20,
                                                                                                differential_privacy=differential_privacy, 
                                                                                                diffusion=diffuse, 
                                                                                                do_prune=False) 
            elif HPO_version == "validation_loss":
                best_params_HIVAE, study = generators_dict[generator_name].optuna_hyperparameter_search_HIVAE_loss(df_init_control_encoded, 
                                                                                                                miss_mask_control, 
                                                                                                                true_miss_mask_control,
                                                                                                                feat_types_dict_ext,
                                                                                                                n_splits=n_splits,
                                                                                                                n_trials=n_trials, 
                                                                                                                generator_name=generator_name, 
                                                                                                                study_name=study_name,
                                                                                                                seed=10, 
                                                                                                                target_epsilon=1.0, # None if not DP, otherwise the target epsilon for the DP generators
                                                                                                                target_delta=1e-5,
                                                                                                                tune_params=None,
                                                                                                                fixed_params={"epochs": 10000, "n_samples_gen": n_samples}, # these parameters will be fixed to the specified value and not tuned,
                                                                                                                norm_mode="global",
                                                                                                                screening_epochs=800,
                                                                                                                n_startup_trials=20,
                                                                                                                differential_privacy=differential_privacy,
                                                                                                                do_prune=False)
                if diffuse:
                    # not implemented for the moment, as it is not the main focus of the paper and would require to re-run all the experiments with the diffuse version
                    raise NotImplementedError("HPO based on validation loss is not implemented yet for the diffuse version of HI-VAE.")
                else:
                    best_params = best_params_HIVAE
            else:
                raise ValueError("Invalid HPO_version. Choose between 'external_metrics' and 'validation_loss'.")                                                                           
            
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
    # generators_sel = ["HI-VAE_weibull", "HI-VAE_piecewise", 
    #                   "Surv-GAN", "Surv-VAE", 
    #                   "HI-VAE_weibull_prior", "HI-VAE_piecewise_prior", 
    #                   "HI-VAE_weibull_DP", "HI-VAE_piecewise_DP", 
    #                   "HI-VAE_weibull_diffuse", "HI-VAE_piecewise_diffuse", 
    #                   "HI-VAE_weibull_diffuse_DP", "HI-VAE_piecewise_diffuse_DP"]
    generators_sel = ["HI-VAE_weibull", "HI-VAE_piecewise"]
    generator_id = int(sys.argv[1])
    run(generators_sel[generator_id])