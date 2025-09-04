import numpy as np
import pandas as pd
import torch

import sys
from pathlib import Path
module_path = Path.cwd().parent / 'utils'
sys.path.append(str(module_path))
import data_processing
from simulations import *
from metrics import general_metrics
module_path = Path.cwd().parent / 'execute'
sys.path.append(str(module_path))
import surv_hivae, surv_gan, surv_vae

import os
import json
import datetime
import uuid

from synthcity.utils.constants import DEVICE
print('Device :', DEVICE)

def adjust_feat_types_for_generator(generator_name, feat_types_dict):
    feat_types_dict_ext = [dict(ft) for ft in feat_types_dict]  # deep copy
    for d in feat_types_dict_ext:
        if d['name'] == "survcens":
            if generator_name == "HI-VAE_weibull" or generator_name == "HI-VAE_weibull_prior":
                d["type"] = 'surv_weibull'
            elif generator_name == "HI-VAE_lognormal":
                d["type"] = 'surv'
            else:
                d["type"] = 'surv_piecewise'
    return feat_types_dict_ext

def run(dataset_name, generators_sel):

    current_path = os.getcwd()  # Get current working directory
    parent_path = os.path.dirname(current_path)
    
    data_file_control_ext = parent_path + "/dataset/" + dataset_name + "/data_control_ext.csv"
    feat_types_file_control_ext = parent_path + "/dataset/" + dataset_name + "/data_types_control_ext.csv"
    data_file_treated_ext = parent_path + "/dataset/" + dataset_name + "/data_treated_ext.csv"
    feat_types_file_treated_ext = parent_path + "/dataset/" + dataset_name + "/data_types_treated_ext.csv"
    data_file_full_ext = parent_path + "/dataset/" + dataset_name + "/data_full_ext.csv"
    feat_types_file_full_ext = parent_path + "/dataset/" + dataset_name + "/data_types_full_ext.csv"

    # If the dataset has no missing data, leave the "miss_file" variable empty
    miss_file = parent_path + "dataset/" + dataset_name + "/Missing.csv"
    true_miss_file = None

    # Load and transform control data
    df_init_control_encoded, feat_types_dict, miss_mask_control, true_miss_mask_control, _ = data_processing.read_data(data_file_control_ext,
                                                                                                                feat_types_file_control_ext,
                                                                                                                miss_file, true_miss_file)
    data_init_control_encoded = torch.from_numpy(df_init_control_encoded.values)
    data_init_control = data_processing.discrete_variables_transformation(data_init_control_encoded, feat_types_dict)

    # Load and transform treated data
    df_init_treated_encoded, _, _, _, _ = data_processing.read_data(data_file_treated_ext, feat_types_file_treated_ext, miss_file, true_miss_file)
    data_init_treated_encoded = torch.from_numpy(df_init_treated_encoded.values)
    data_init_treated = data_processing.discrete_variables_transformation(data_init_treated_encoded, feat_types_dict)

    # Load and transform full data
    df_init_full_encoded, feat_types_dict_full, miss_mask_full, true_miss_mask_full, _ = data_processing.read_data(data_file_full_ext, 
                                                                                                                   feat_types_file_full_ext, 
                                                                                                                   miss_file, true_miss_file)

    data_init_full_encoded = torch.from_numpy(df_init_full_encoded.values)
    data_init_full = data_processing.discrete_variables_transformation(data_init_full_encoded, feat_types_dict_full)

    fnames = ['time', 'censor'] + pd.read_csv(feat_types_file_control_ext)["name"].to_list()[1:]

    # Format data in dataframe
    df_init_treated = pd.DataFrame(data_init_treated.numpy(), columns=fnames)
    df_init_control = pd.DataFrame(data_init_control.numpy(), columns=fnames)
    df_init_full = pd.concat([df_init_control, df_init_treated], ignore_index=True)

    # Parameters of the optuna study
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
    best_param_dir = parent_path + "/dataset/" + dataset_name + "/optuna_results"
    best_params_dict = {}
    for generator_name in generators_sel:
        # best_param_file = [item for item in best_param_files if generator_name in item][0]
        for f in os.listdir(best_param_dir):
            if (f.endswith(generator_name + '.json') & ("trainfull_" + name_config in f)):
                best_param_file = f
        with open(best_param_dir + "/" + best_param_file, "r") as f:
            best_params_dict[generator_name] = json.load(f)

    os.chdir(work_dir)

    df_gen_control_dict ={}
    df_syn_dict = {}
    # For each generator, perform the data generation with the best params
    for generator_name in generators_sel:
        best_params = best_params_dict[generator_name]
        if generator_name in ["HI-VAE_lognormal", "HI-VAE_weibull", "HI-VAE_piecewise", "HI-VAE_weibull_prior", "HI-VAE_piecewise_prior"]:
            if generator_name in ["HI-VAE_weibull_prior", "HI-VAE_piecewise_prior"]:
                gen_from_prior = True
            else:
                gen_from_prior = False
            feat_types_dict_ext = adjust_feat_types_for_generator(generator_name, feat_types_dict)
            condition = {'var': 'treatment_0', 'value': 1.0, 'n_samples': df_init_control.shape[0]}  # Condition on the control group
            data_gen_control = generators_dict[generator_name].run(df_init_full_encoded, miss_mask_full, 
                                                                    true_miss_mask_full, feat_types_dict_ext, 
                                                                    n_generated_dataset, params=best_params, 
                                                                    epochs=10000, gen_from_prior=gen_from_prior,
                                                                    condition=condition)
        elif generator_name in ["Surv-VAE"]:
            condition = {'var': 'treatment', 'value': 0.0, 'n_samples': df_init_control.shape[0]}
            data_gen_control = generators_dict[generator_name].run(data_init_full, columns=fnames, 
                                                                    target_column="censor", time_to_event_column="time", 
                                                                    n_generated_dataset=n_generated_dataset, 
                                                                    params=best_params, condition=condition)
        else:
            cond_gen = df_init_control[["censor", "treatment"]]
            data_gen_control = generators_dict[generator_name].run(data_init_full, columns=fnames, 
                                                                    target_column="censor", time_to_event_column="time", 
                                                                    n_generated_dataset=n_generated_dataset, 
                                                                    n_generated_sample=len(cond_gen),
                                                                    params=best_params, cond_gen=cond_gen)

        list_df_gen_control = []
        data_syn = []
        for i in range(n_generated_dataset):
            df_gen_control = pd.DataFrame(data_gen_control[i].numpy(), columns=fnames)
            df_gen_control["treatment"] = 0
            list_df_gen_control.append(df_gen_control)
            data_syn.append(pd.concat([df_init_treated, df_gen_control], ignore_index=True))
        df_gen_control_dict[generator_name] = list_df_gen_control
        df_syn_dict[generator_name] = data_syn

    if not os.path.exists(parent_path + "/dataset/" + dataset_name + "/metric_results"):
        os.makedirs(parent_path + "/dataset/" + dataset_name + "/metric_results")

    general_scores = []
    for generator_name in generators_sel:
        general_scores.append(general_metrics(df_init_control, df_gen_control_dict[generator_name], generator_name))
    general_scores_df = pd.concat(general_scores)
    general_scores_df.to_csv(parent_path + "/dataset/" + dataset_name + '/metric_results/trainfull_general_scores_df.csv', index=False)

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
    generators_sel = ["HI-VAE_weibull", "HI-VAE_piecewise", "Surv-GAN", "Surv-VAE", "HI-VAE_weibull_prior", "HI-VAE_piecewise_prior"]
    dataset_sel = ["Aids", "SAS_1", "SAS_2", "SAS_3"]
    dataset_id = int(sys.argv[1])
    dataset_name = dataset_sel[dataset_id]
    run(dataset_name , generators_sel)