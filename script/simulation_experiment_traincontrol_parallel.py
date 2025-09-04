import numpy as np
import pandas as pd
import torch

import sys
from pathlib import Path
module_path = Path.cwd().parent / 'utils'
sys.path.append(str(module_path))
import data_processing
from simulations import *
from metrics import fit_cox_model, general_metrics
module_path = Path.cwd().parent / 'execute'
sys.path.append(str(module_path))
import surv_hivae, surv_gan, surv_vae
from sksurv.nonparametric import kaplan_meier_estimator

import os
import uuid
import datetime
import json
import sys

from synthcity.utils.constants import DEVICE
print('Device :', DEVICE)

def true_univ_coef(treatment_effect, independent = True, feature_types_list = ["pos", "real", "cat"],
                   n_features_bytype = 4, n_active_features = 3 , p_treated = 0.5, shape_T = 2, shape_C = 2,
                   scale_C = 6., scale_C_indep = 2.5, data_types_create = True, seed=0):

    """Compute the univariate treatment effect from large sample simulated data."""
    n_samples = 100000
    seed = int(np.random.randint(1000))
    control, treated, _ = simulation(treatment_effect, n_samples,
                                     independent=independent,
                                     n_features_bytype=n_features_bytype,
                                     n_active_features=n_active_features,
                                     feature_types_list=feature_types_list,
                                     shape_T=shape_T, shape_C=shape_C,
                                     scale_C=scale_C, scale_C_indep=scale_C_indep, seed=seed)

    df_init = pd.concat([control, treated], ignore_index=True)
    columns = ['time', 'censor', 'treatment']
    coef_init = fit_cox_model(df_init, columns)[0]
    return coef_init[0]

def prepare_dataset_dirs(parent_path, dataset_name):
    base_path = os.path.join(parent_path + "/dataset", dataset_name)
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(os.path.join(base_path, "optuna_results"), exist_ok=True)
    return base_path

def save_parameters(param_path, params):
    with open(param_path, "w") as f:
        for key, value in params.items():
            f.write(f"{key} = {value}\n")

def kaplan_meier_estimation(surv_data, label=None, ax=None):
    """Plot Kaplan-Meier curve with confidence interval."""
    surv_time = surv_data['time'].values
    surv_event = surv_data['censor'].values.astype(bool)
    uniq_time, surv_prob, conf_int = kaplan_meier_estimator(surv_event, surv_time, conf_type="log-log")

    ax.step(uniq_time, surv_prob, where="post", label=label)
    ax.fill_between(uniq_time, conf_int[0], conf_int[1], alpha=0.25, step="post")


def adjust_feat_types_for_generator(generator_name, feat_types_dict):
    feat_types_dict_ext = [dict(ft) for ft in feat_types_dict]  # deep copy
    for d in feat_types_dict_ext:
        if d['name'] == "survcens":
            if "HI-VAE_weibull" in generator_name: # == "HI-VAE_weibull" or generator_name == "HI-VAE_weibull_prior":
                d["type"] = 'surv_weibull'
            elif generator_name == "HI-VAE_lognormal":
                d["type"] = 'surv'
            else:
                d["type"] = 'surv_piecewise'
    return feat_types_dict_ext

def setup_unique_working_dir(base_dir="experiments"):
    original_dir = os.getcwd()  # Save original dir
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    uid = uuid.uuid4().hex[:8]
    work_dir = os.path.join(base_dir, f"run_{timestamp}_{uid}")
    os.makedirs(work_dir, exist_ok=True)
    return original_dir, work_dir  # Return the original dir

def run(MC_id):

    # Simulation parameters
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


    metric_optuna = "survival_km_distance"
    dataset_name = "Simulations_indep_traincontrol_DP"
    current_path = os.getcwd()  # Get current working directory
    parent_path = os.path.dirname(current_path)
    base_path = prepare_dataset_dirs(parent_path, dataset_name)
    param_file = os.path.join(base_path, "params.txt")
    save_parameters(param_file, {
        "n_samples": n_samples,
        "n_features_bytype": n_features_bytype,
        "n_active_features": n_active_features,
        "treatment_effect": treatment_effect,
        "p_treated": p_treated,
        "shape_T": shape_T,
        "shape_C": shape_C,
        "scale_C": scale_C,
        "scale_C_indep": scale_C_indep,
        "feature_types_list": feature_types_list,
        "independent": independent,
        "data_types_create": data_types_create
    })
    generators_sel = ["HI-VAE_weibull", "HI-VAE_piecewise", "Surv-GAN", "Surv-VAE", "HI-VAE_weibull_prior", "HI-VAE_piecewise_prior"]
    generators_dict = {"HI-VAE_weibull" : surv_hivae,
                       "HI-VAE_piecewise" : surv_hivae,
                       "HI-VAE_lognormal" : surv_hivae,
                       "Surv-GAN" : surv_gan,
                       "Surv-VAE" : surv_vae, 
                       "HI-VAE_weibull_prior" : surv_hivae, 
                       "HI-VAE_piecewise_prior" : surv_hivae,
                       "HI-VAE_weibull_DP" : surv_hivae, 
                       "HI-VAE_piecewise_DP" : surv_hivae}

    # BEST PARAMETERS
    best_params_dict = {}
    name_config = "simu_N{}_nfeat{}_t{}".format(n_samples, n_features_bytype, int(treatment_effect))
    n_trials = 150
    for generator_name in generators_sel:
        # n_trials = min(100, int(multiplier_trial * generators_dict[generator_name].get_n_hyperparameters(generator_name)))
        best_params_file = os.path.join(base_path, "optuna_results", "best_params_{}_ntrials{}_{}_{}.json".format(name_config, n_trials, metric_optuna, generator_name))
        with open(best_params_file, "r") as f:
            best_params_dict[generator_name] = json.load(f)

    
    # MONTE-CARLO EXPERIMENT
    n_MC_exp = 10
    treat_effects = np.arange(0., 1.1, 0.2)
    n_generated_dataset = 200
    synthcity_metrics_sel = ['J-S distance', 'KS test', 'Survival curves distance',
                                'Detection XGB', 'NNDR', 'K-map score']

    # Initialize storage for metrics and results
    synthcity_metrics_res_dict = {generator_name: pd.DataFrame() for generator_name in generators_sel}
    log_p_value_gen_dict = {generator_name: [] for generator_name in generators_sel}
    log_p_value_control_dict = {generator_name: [] for generator_name in generators_sel}
    est_cox_coef_gen_dict = {generator_name: [] for generator_name in generators_sel}
    est_cox_coef_se_gen_dict = {generator_name: [] for generator_name in generators_sel}
    
    # Initialize result variables for MC experiment
    simu_num = []
    D_control = []
    D_treated = []
    coef_init_univ_list = []
    H0_coef = []
    log_p_value_init = []
    est_cox_coef_init = []
    est_cox_coef_se_init = []

    # Seed
    seed = MC_id * n_MC_exp # 0, 10, 20, .. 
    print("Run Monte Carlo experiments {} to {}...".format(MC_id * n_MC_exp + 1, (MC_id + 1) * n_MC_exp))
    dataset_name_MC = dataset_name + "/MC_{}to{}".format(MC_id * n_MC_exp + 1, (MC_id + 1) * n_MC_exp) 
    if not os.path.exists(parent_path + "/dataset/" + dataset_name_MC):
        os.makedirs(parent_path + "/dataset/" + dataset_name_MC)

    # Set a unique working directory for this job
    original_dir, work_dir = setup_unique_working_dir("parallel_runs")
    os.chdir(work_dir)  # Switch to private work dir
    print("Working directory:", work_dir)
    print("Original directory:", original_dir)
    
    for m in np.arange(n_MC_exp):
        print("Monte-Carlo experiment", m + n_MC_exp * MC_id)

        # To make sure the difference between simulated datasets, increase seed value each time
        seed += 1

        # Simulate control group data
        treatment_effect = 0.0
        control, _, types = simulation(treatment_effect, n_samples, independent, feature_types_list,
                                       n_features_bytype, n_active_features, p_treated, shape_T,
                                       shape_C, scale_C, scale_C_indep, data_types_create, seed=seed)
        control = control.drop(columns='treatment')

        data_file_control = os.path.join(f"{parent_path}/dataset/{dataset_name_MC}", "data_control.csv")
        feat_types_file_control = os.path.join(f"{parent_path}/dataset/{dataset_name_MC}", "data_types_control.csv")
        control.to_csv(data_file_control, index=False, header=False)
        types.to_csv(feat_types_file_control, index=False)
        
        # Load and process control data
        df_init_control_encoded, feat_types_dict, miss_mask_control, true_miss_mask_control, _ = data_processing.read_data(
                data_file_control, feat_types_file_control, miss_file="Missing.csv", true_miss_file=None)
        data_init_control_encoded = torch.from_numpy(df_init_control_encoded.values)
        data_init_control = data_processing.discrete_variables_transformation(data_init_control_encoded, feat_types_dict)

        # Format control data into DataFrame
        fnames = types['name'][:-1].tolist() + ["time", "censor"]
        df_init_control = pd.DataFrame(data_init_control.numpy(), columns=fnames)
        df_init_control["treatment"] = 0

        df_gen_control_dict ={}
        # For each generator, perform the data generation with the best params
        for generator_name in generators_sel:
            best_params = best_params_dict[generator_name]
            if generator_name in ["HI-VAE_lognormal", "HI-VAE_weibull", "HI-VAE_piecewise", "HI-VAE_weibull_prior", "HI-VAE_piecewise_prior",  "HI-VAE_weibull_DP", "HI-VAE_piecewise_DP"]:
                if "_prior" in generator_name:
                    gen_from_prior = True
                else:
                    gen_from_prior = False
                if "_DP" in generator_name:
                    differential_privacy = True
                else:
                    differential_privacy = False
                feat_types_dict_ext = adjust_feat_types_for_generator(generator_name, feat_types_dict)
                data_gen_control = generators_dict[generator_name].run(df_init_control_encoded, miss_mask_control, 
                                                                       true_miss_mask_control, feat_types_dict_ext, 
                                                                       n_generated_dataset, params=best_params, 
                                                                       epochs=10000, gen_from_prior=gen_from_prior, 
                                                                       differential_privacy=differential_privacy,
                                                                       batchcorrect=True)
            else:
                data_gen_control = generators_dict[generator_name].run(data_init_control, columns=fnames, 
                                                                       target_column="censor", time_to_event_column="time", 
                                                                       n_generated_dataset=n_generated_dataset, 
                                                                       params=best_params)

            list_df_gen_control = []
            for i in range(n_generated_dataset):
                df_gen_control = pd.DataFrame(data_gen_control[i].numpy(), columns=fnames)
                df_gen_control["treatment"] = 0
                list_df_gen_control.append(df_gen_control)
            df_gen_control_dict[generator_name] = list_df_gen_control

            # Save generated data and compute metrics
            synthcity_metrics_res = general_metrics(df_init_control, list_df_gen_control, generator_name)[synthcity_metrics_sel]
            synthcity_metrics_res_ext = pd.concat([synthcity_metrics_res] * len(treat_effects))
            synthcity_metrics_res_dict[generator_name] = pd.concat([synthcity_metrics_res_dict[generator_name], 
                                                                    synthcity_metrics_res_ext])


        # Compare the performance of generation in terms of p-values between generated control and treated group
        for t, treatment_effect in enumerate(treat_effects):
            coef_init_univ = true_univ_coef(treatment_effect, independent, feature_types_list,
                                            n_features_bytype, n_active_features, p_treated, shape_T,
                                            shape_C, scale_C, scale_C_indep, data_types_create, seed=seed)

            _, treated, _ = simulation(treatment_effect, n_samples, independent, feature_types_list,
                                       n_features_bytype, n_active_features, p_treated, shape_T,
                                       shape_C, scale_C, scale_C_indep, data_types_create, seed=seed)
            treated = treated.drop(columns='treatment')

            data_file_treated = os.path.join(f"{parent_path}/dataset/{dataset_name_MC}", "data_treated.csv")
            feat_types_file_treated = os.path.join(f"{parent_path}/dataset/{dataset_name_MC}", "data_types_treated.csv")
            treated.to_csv(data_file_treated, index=False, header=False)
            types.to_csv(feat_types_file_treated, index=False)

            # Load and process treated data
            df_init_treated_encoded, _, _, _, _ = data_processing.read_data(data_file_treated, feat_types_file_treated, 
                                                                            miss_file="Missing.csv", true_miss_file=None)
            data_init_treated_encoded = torch.from_numpy(df_init_treated_encoded.values)
            data_init_treated = data_processing.discrete_variables_transformation(data_init_treated_encoded, feat_types_dict)
            df_init_treated = pd.DataFrame(data_init_treated.numpy(), columns=fnames)
            df_init_treated["treatment"] = 1

            # Combine control and treated data
            df_init = pd.concat([df_init_control, df_init_treated], ignore_index=True)
            columns = ['time', 'censor', 'treatment']
            coef_init, _, _, se_init = fit_cox_model(df_init, columns)
            est_cox_coef_init += [coef_init[0]] * n_generated_dataset
            est_cox_coef_se_init += [se_init[0]] * n_generated_dataset

            # Compute log-rank test p-value for initial control group vs initial treated group
            p_value_init = compute_logrank_test(df_init_control, df_init_treated)
            log_p_value_init += [p_value_init] * n_generated_dataset
            H0_coef += [treatment_effect] * n_generated_dataset
            simu_num += [(m + n_MC_exp * MC_id) * len(treat_effects) + t] * n_generated_dataset
            D_control += [control['censor'].sum()] * n_generated_dataset
            D_treated += [treated['censor'].sum()] * n_generated_dataset
            coef_init_univ_list += [coef_init_univ] * n_generated_dataset

            # For each generator, compute the log-rank test p-values and Cox coefficients for generated control group vs initial treated group
            for generator_name in generators_sel:
                log_p_value_gen_list = []
                log_p_value_control_list = []
                est_cox_coef_gen = []
                est_cox_coef_se_gen = []
                for i in range(n_generated_dataset):
                    df_gen_control = df_gen_control_dict[generator_name][i]
                    log_p_value_gen_list.append(compute_logrank_test(df_gen_control, treated))
                    log_p_value_control_list.append(compute_logrank_test(df_gen_control, control))
                    df_gen = pd.concat([df_gen_control, df_init_treated], ignore_index=True)
                    coef_gen, _, _, se_gen = fit_cox_model(df_gen, columns)
                    est_cox_coef_gen.append(coef_gen[0])
                    est_cox_coef_se_gen.append(se_gen[0])

                log_p_value_gen_dict[generator_name] += log_p_value_gen_list
                log_p_value_control_dict[generator_name] += log_p_value_control_list
                est_cox_coef_gen_dict[generator_name] += est_cox_coef_gen
                est_cox_coef_se_gen_dict[generator_name] += est_cox_coef_se_gen


    # Save the results
    results = pd.DataFrame({'XP_num': simu_num, 'D_control': D_control, 'D_treated': D_treated,
                            'H0_coef_univ': coef_init_univ_list, 'H0_coef': H0_coef,
                            'log_pvalue_init': log_p_value_init, 'est_cox_coef_init': est_cox_coef_init,
                            'est_cox_coef_se_init': est_cox_coef_se_init})

    # Add metrics and coefficients for each generator
    for generator_name in generators_sel:
        results[f"log_pvalue_{generator_name}"] = log_p_value_gen_dict[generator_name]
        results[f"log_pvalue_control_{generator_name}"] = log_p_value_control_dict[generator_name]
        results[f"est_cox_coef_{generator_name}"] = est_cox_coef_gen_dict[generator_name]
        results[f"est_cox_coef_se_{generator_name}"] = est_cox_coef_se_gen_dict[generator_name]
        for metric in synthcity_metrics_sel:
            results[f"{metric}_{generator_name}"] = synthcity_metrics_res_dict[generator_name][metric].values

    MC_init = MC_id * n_MC_exp + 1
    MC_final = (MC_id + 1) * n_MC_exp
    results.to_csv(f"{parent_path}/dataset/{dataset_name}/results_{metric_optuna}_n_samples_{n_samples}_n_features_bytype_{n_features_bytype}_MC_{MC_init}to{MC_final}.csv")

if __name__ == "__main__":
    MC_id = int(sys.argv[1])
    run(MC_id)
    