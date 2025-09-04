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

import os

def run(dataset_name):

    current_path = os.getcwd()  # Get current working directory
    parent_path = os.path.dirname(current_path)
     ## DATA LOADING
    data_file_control= parent_path + "/dataset/" + dataset_name + "/data_control.csv"
    feat_types_file_control =parent_path + "/dataset/" + dataset_name + "/data_types_control.csv"
    data_file_treated= parent_path + "/dataset/" + dataset_name + "/data_treated.csv"
    feat_types_file_treated= parent_path + "/dataset/" + dataset_name + "/data_types_treated.csv"

    # If the dataset has no missing data, leave the "miss_file" variable empty
    m_perc = 10
    mask = 1
    miss_file = parent_path + "/dataset/" + dataset_name + "/Missing{}_{}.csv".format(m_perc, mask)
    true_miss_file = None

    fnames = ['time', 'censor'] + pd.read_csv(feat_types_file_control)["name"].to_list()[1:]
    # Load and transform control data
    df_init_control_encoded, feat_types_dict, _, _, _ = data_processing.read_data(data_file_control, feat_types_file_control, miss_file, true_miss_file)
    data_init_control_encoded = torch.from_numpy(df_init_control_encoded.values)
    data_init_control = data_processing.discrete_variables_transformation(data_init_control_encoded, feat_types_dict)
    df_init_control = pd.DataFrame(data_init_control.numpy(), columns=fnames)
    df_init_control['treatment'] = 0.0
    
    # Load and transform treated data
    df_init_treated_encoded, _, _, _, _ = data_processing.read_data(data_file_treated, feat_types_file_treated, miss_file, true_miss_file)
    data_init_treated_encoded = torch.from_numpy(df_init_treated_encoded.values)
    data_init_treated = data_processing.discrete_variables_transformation(data_init_treated_encoded, feat_types_dict)
    df_init_treated = pd.DataFrame(data_init_treated.numpy(), columns=fnames)
    df_init_treated['treatment'] = 1.0

    # Add treatment column
    new_row = {'name': 'treatment', 'type': 'cat', 'dim': 1, 'nclass': 2}
    df_types = pd.read_csv(feat_types_file_control)
    types = pd.concat([df_types, pd.DataFrame([new_row])], ignore_index=True)
    types['nclass'] = types['nclass'].astype('Int64')

    df_init_full = pd.concat([df_init_control, df_init_treated], axis=0, ignore_index=True)
    df_init_full['treatment'] = df_init_full['treatment'].astype(np.float32)

    if not os.path.exists(parent_path + "/dataset"):
        os.makedirs(parent_path + "/dataset/")

    # Save the data
    if not os.path.exists(parent_path + "/dataset/" + dataset_name):
        os.makedirs(parent_path + "/dataset/" + dataset_name)
    
    data_file_control_ext = parent_path + "/dataset/" + dataset_name + "/data_control_ext.csv"
    feat_types_file_control_ext = parent_path + "/dataset/" + dataset_name + "/data_types_control_ext.csv"
    data_file_treated_ext = parent_path + "/dataset/" + dataset_name + "/data_treated_ext.csv"
    feat_types_file_treated_ext = parent_path + "/dataset/" + dataset_name + "/data_types_treated_ext.csv"
    data_file_full_ext = parent_path + "/dataset/" + dataset_name + "/data_full_ext.csv"
    feat_types_file_full_ext = parent_path + "/dataset/" + dataset_name + "/data_types_full_ext.csv"

    # If the dataset has no missing data, leave the "miss_file" variable empty
    miss_file = parent_path + "/dataset/" + dataset_name + "/Missing.csv"
    true_miss_file = None

    df_init_control.to_csv(data_file_control_ext, index=False, header=False)
    types.to_csv(feat_types_file_control_ext, index=False)
    df_init_treated.to_csv(data_file_treated_ext, index=False, header=False)
    types.to_csv(feat_types_file_treated_ext, index=False)
    df_init_full.to_csv(data_file_full_ext, index=False, header=False)
    types.to_csv(feat_types_file_full_ext, index=False)

  

if __name__ == "__main__":
    dataset_names = ["Aids", "SAS_1", "SAS_2", "SAS_3"]
    for dataset_name in dataset_names:
        run(dataset_name)