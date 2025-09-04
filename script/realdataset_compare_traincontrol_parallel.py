import numpy as np
import pandas as pd
import torch

import sys
from pathlib import Path
module_path = Path.cwd().parent / 'utils'
sys.path.append(str(module_path))
import data_processing
from simulations import *
from metrics import general_metrics, replicability_ext, fit_cox_model
module_path = Path.cwd().parent / 'execute'
sys.path.append(str(module_path))
import surv_hivae, surv_gan, surv_vae

import os
import json
import datetime
import uuid

from synthcity.utils.constants import DEVICE
print('Device :', DEVICE)

def run(dataset_name, generators_sel):

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
            if (f.endswith(generator_name + '.json') & ("traincontrol_" + name_config in f)):
                best_param_file = f
        with open(best_param_dir + "/" + best_param_file, "r") as f:
            best_params_dict[generator_name] = json.load(f)

    os.chdir(work_dir)
    data_gen_control_dict = {}
    for generator_name in generators_sel:
        print("=" * 100)
        print("Generate data by " + generator_name)
        best_params = best_params_dict[generator_name]
        if generator_name in ["HI-VAE_weibull", "HI-VAE_piecewise", "HI-VAE_weibull_prior", "HI-VAE_piecewise_prior"]:
            feat_types_dict_ext = feat_types_dict.copy()
            for i in range(len(feat_types_dict)):
                if feat_types_dict_ext[i]['name'] == "survcens":
                    if generator_name in["HI-VAE_weibull", "HI-VAE_weibull_prior"]:
                        feat_types_dict_ext[i]["type"] = 'surv_weibull'
                    else:
                        feat_types_dict_ext[i]["type"] = 'surv_piecewise'
            if generator_name in ["HI-VAE_weibull_prior", "HI-VAE_piecewise_prior"]:
                gen_from_prior = True
            else:
                gen_from_prior = False
        if generator_name in ["HI-VAE_weibull", "HI-VAE_piecewise", "HI-VAE_weibull_prior", "HI-VAE_piecewise_prior"]:
            data_gen_control_dict[generator_name] = generators_dict[generator_name].run(df_init_control_encoded, miss_mask_control, true_miss_mask_control, feat_types_dict_ext, n_generated_dataset, params=best_params, epochs = 10000,
                                                                                        gen_from_prior=gen_from_prior)
        else:
            data_gen_control_dict[generator_name] = generators_dict[generator_name].run(data_init_control, columns=fnames, target_column="censor", time_to_event_column="time", n_generated_dataset=n_generated_dataset, params=best_params)

    # Convert generated data into dataframe
    df_gen_control_dict = {}
    df_syn_dict = {}
    for generator_name in generators_sel:
        list_df_gen_control = []
        data_syn = []
        for j in range(n_generated_dataset):
            df_gen_control_j = pd.DataFrame(data_gen_control_dict[generator_name][j].numpy(), columns=fnames)
            df_gen_control_j['treatment'] = 0
            list_df_gen_control.append(df_gen_control_j)
            data_syn.append(pd.concat([df_init_treated, df_gen_control_j], ignore_index=True))

        df_gen_control_dict[generator_name] = list_df_gen_control
        df_syn_dict[generator_name] = data_syn

    if not os.path.exists(parent_path + "/dataset/" + dataset_name + "/metric_results"):
        os.makedirs(parent_path + "/dataset/" + dataset_name + "/metric_results")

    general_scores = []
    for generator_name in generators_sel:
        general_scores.append(general_metrics(df_init_control, df_gen_control_dict[generator_name], generator_name))
    general_scores_df = pd.concat(general_scores)
    general_scores_df.to_csv(parent_path + "/dataset/" + dataset_name + '/metric_results/traincontrol_general_scores_df.csv', index=False)

    replicability_scores = []
    for generator_name in generators_sel:
        replicability_scores.append(replicability_ext(df_init, df_syn_dict[generator_name], generator_name))
    replicability_scores_df = pd.concat(replicability_scores, ignore_index=True)
    replicability_scores_df.to_csv(parent_path + "/dataset/" + dataset_name + '/metric_results/traincontrol_replicability_scores_df.csv', index=False)

    columns = ['time', 'censor', 'treatment']
    _, _, ci_init, _ = fit_cox_model(df_init, columns)
    # Compute midpoints and widths
    midpoints = [(ci_init[1] + ci_init[0]) / 2]
    errors = [(ci_init[1] - ci_init[0]) / 2]
    label = ["Init"]
    colors = ['green', 'blue', 'orange', 'cyan', 'magenta', 'grey']

    colors_ = ['red']
    for i , generator in enumerate(generators_sel):
        data_syn_ = df_syn_dict[generator]
        results = [fit_cox_model(data, columns) for data in data_syn_]
        coef_syn, _, _, se_syn = zip(*results)
        for n in range(10):
            coef_syn_, se_syn_ = np.array(coef_syn)[n][0], np.array(se_syn)[n][0]
            ci_syn = (coef_syn_ - 1.96 * se_syn_, coef_syn_ + 1.96 * se_syn_)
            midpoints.append((ci_syn[1] + ci_syn[0]) / 2)
            errors.append((ci_syn[1] - ci_syn[0]) / 2)
            label.append(generator + " " + str(n + 1))
            colors_.append(colors[i])

    err_df = pd.DataFrame({"midpoints" : midpoints,
                          "errors" : errors,
                          "label" : label,
                          "colors" : colors_})
    err_df.to_csv(parent_path + "/dataset/" + dataset_name + '/metric_results/traincontrol_error_df.csv', index=False)

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