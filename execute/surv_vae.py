import pandas as pd
from synthcity.plugins.core.dataloader import SurvivalAnalysisDataLoader
from synthcity.plugins import Plugins
from synthcity.utils.optuna_sample import suggest_all
from synthcity.utils.reproducibility import clear_cache, enable_reproducible_results
from synthcity.metrics.eval import Metrics
from sklearn.model_selection import KFold
import sys
from pathlib import Path
module_path = Path.cwd().parent / 'utils'
sys.path.append(str(module_path))
import metrics, data_processing
import numpy as np
import optuna
import os
import random
import torch

import multiprocessing as mp
mp.set_start_method("spawn", force=True)

def run_worker(return_dict, model, params, data, count):
    # print("training....")
    model_trial = model(**params)
    model_trial.fit(data)
    # print("generation....")
    result = model_trial.generate(count=count)
    return_dict["result"] = result

def run_with_timeout_mp(model, params, data, count, timeout=60):
    manager = mp.Manager()
    return_dict = manager.dict()
    p = mp.Process(target=run_worker, args=(return_dict, model, params, data, count))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        print(f"Generation timed out after {timeout} seconds.")
        raise optuna.TrialPruned()
    
    return return_dict["result"]

def set_seed(seed=1):
    random.seed(seed)                            # Python built-in
    np.random.seed(seed)                         # NumPy
    torch.manual_seed(seed)                      # PyTorch (CPU)


def generate_survae(model, n_generated_dataset, n_generated_sample, target_column, time_to_event_column, condition=None):
    
    est_data_gen_transformed_survae = []

    if condition is None:
        for j in range(n_generated_dataset):
            out = model.generate(count=n_generated_sample)
            est_data_gen_transformed_survae.append(out)
    
    else:
        min_shape = 0
        i = 0
        while min_shape < condition['n_samples']:

            if i > 0:
                est_data_gen_transformed_survae = [df[:min_shape] for df in est_data_gen_transformed_survae]

            for j in range(n_generated_dataset):
                out = model.generate(count=n_generated_sample)
                out_df = out.dataframe()
                out_df = out_df[out_df[condition['var']] == condition['value']]
                if i == 0:
                    est_data_gen_transformed_survae.append(out_df)
                else:   
                    est_data_gen_transformed_survae[j] = pd.concat([est_data_gen_transformed_survae[j], out_df], ignore_index=True)

            shapes = [len(t) for t in est_data_gen_transformed_survae]
            min_shape = min(shapes)
            i += 1
        est_data_gen_transformed_survae = [df[:condition['n_samples']] for df in est_data_gen_transformed_survae] 
        est_data_gen_transformed_survae = [SurvivalAnalysisDataLoader(df, target_column=target_column, time_to_event_column=time_to_event_column) for df in est_data_gen_transformed_survae] 

    return est_data_gen_transformed_survae


def run(data, columns, target_column, time_to_event_column, n_generated_dataset, n_generated_sample=None, params=None, condition=None, apply_rounding=False, feat_types_dict=None):
    # condition={'var': "treatment", 'value': 0.0, 'n_samples': 300}
    """
    Use a VAE for tabular data generation
    """

    set_seed()

    # Define data and model
    df = pd.DataFrame(data.numpy(), columns=columns) # Preprocessed dataset
    data = SurvivalAnalysisDataLoader(df, target_column=target_column, time_to_event_column=time_to_event_column)
    
    if params is not None:
        model = type(Plugins().get("survae"))
        model_survae = model(**params)
    else:
        model_survae = Plugins().get("survae") 
    
    # Train
    model_survae.fit(data)
    
    # Generate
    if isinstance(n_generated_sample, list):
        est_data_gen_transformed_survae_list = []
        for n_generated_sample_ in n_generated_sample:
            est_data_gen_transformed_survae = generate_survae(model_survae, n_generated_dataset, n_generated_sample_, target_column, time_to_event_column, condition)
            est_data_gen_transformed_survae_list.append(est_data_gen_transformed_survae)

        return est_data_gen_transformed_survae_list
    else:
        # TODO; Apply rounding only here
        if n_generated_sample is None:
            n_generated_sample = data.shape[0]
        est_data_gen_transformed_survae = generate_survae(model_survae, n_generated_dataset, n_generated_sample, target_column, time_to_event_column, condition)
        est_data_gen_transformed_survae_rounded = []
        if apply_rounding:
            for j in range(n_generated_dataset):
                est_data_gen_transformed_survae_rounded_j = data_processing.round_data_gen(df.values, torch.from_numpy(est_data_gen_transformed_survae[j].numpy()), feat_types_dict)
                est_data_gen_transformed_survae_rounded.append(est_data_gen_transformed_survae_rounded_j)
        return est_data_gen_transformed_survae_rounded
    
def _evaluate_generation(gen_data, ref_loader, metrics_list):
    metrics_dict_evaluation, metrics_synthcity, expected_metrics = metrics.map_metrics_HPO(metrics_list)
    clear_cache()
    evaluation = Metrics().evaluate(
        X_gt=ref_loader,
        X_syn=gen_data,
        reduction='mean',
        n_histogram_bins=10,
        # n_folds=1,
        metrics=metrics_dict_evaluation,
        task_type='survival_analysis',
        use_cache=True,
    )
    scores = []
    for metric in metrics_synthcity:
        if metric in evaluation.T.columns:
            if expected_metrics[metric] == "max":
                val = - evaluation.T[[metric]].T["mean"].values[0]
            else:
                val = evaluation.T[[metric]].T["mean"].values[0]
            print(f"{metric}: {val:.4f}")
        else:
            val = np.nan
            print(f"Warning: metric '{metric}' not found in evaluation results. Using NaN as fallback.")
        scores.append(val)
    return scores
   

def optuna_hyperparameter_search(data, columns, target_column, time_to_event_column, n_generated_dataset, n_splits, n_trials,
                                 n_generated_sample=None, study_name='optuna_study_survae', metric='survival_km_distance',
                                 method='', condition=None, cond_df=None, seed=10, apply_rounding=False, feat_types_dict=None):
    
    df = pd.DataFrame(data.numpy(), columns=columns) # Preprocessed dataset
    if condition is not None and cond_df is not None:
        cond_full_data_loader =  SurvivalAnalysisDataLoader(cond_df, target_column = "censor", time_to_event_column = "time")

    metrics_list = metric if isinstance(metric, list) else [metric]
    print(f"Metrics for optimization: {metrics_list}")
    is_multi_objective = len(metrics_list) > 1
 
    def objective(trial: optuna.Trial):
        set_seed()
        model_survae = type(Plugins().get("survae"))
        hp_space = model_survae.hyperparameter_space()
        # hp_space[0].high = 100  # speed up for now
        hp_space[3].choices = [1e-3, 1e-4, 1e-5]
        hp_space[4].choices = [64, 128, 200, 256, 512]
        params = suggest_all(trial, hp_space)
        ID = f"trial_{trial.number}"
        print(ID)
        scores = []
        try:
            if method == 'train_full_gen_full':
                full_data_loader = SurvivalAnalysisDataLoader(df, target_column=target_column, time_to_event_column=time_to_event_column)
                # model_survae_trial = model_survae(**params)
                # # train on full data
                # model_survae_trial.fit(full_data_loader)
            
                if condition is None:
                    n_gen_sample = n_generated_sample if n_generated_sample is not None else data.shape[0]
                    
                    # gen_data = model_survae_trial.generate(count=n_gen_sample*n_generated_dataset)
                    # clear_cache()

                    gen_data = run_with_timeout_mp(model_survae, params, full_data_loader, n_gen_sample*n_generated_dataset, timeout=120)
                    if apply_rounding:
                        out_rounded = data_processing.round_data_gen(df.values, torch.from_numpy(gen_data.numpy()), feat_types_dict)
                        out_df = pd.DataFrame(out_rounded, columns=columns) # Preprocessed dataset
                        gen_data = SurvivalAnalysisDataLoader(out_df, target_column=target_column, time_to_event_column=time_to_event_column)
                    scores = _evaluate_generation(gen_data, full_data_loader, metrics_list)
                else:
                    est_data_gen_transformed_survae = []
                    gen_shape = 0
                    i = 0
                    while gen_shape < condition['n_samples']*n_generated_dataset:
                        # out = model_survae_trial.generate(count=df.shape[0]*n_generated_dataset)
                        out = run_with_timeout_mp(model_survae, params, full_data_loader, df.shape[0]*n_generated_dataset, timeout=120)  
                        out_df = out.dataframe()
                        out_df = out_df[out_df[condition['var']] == condition['value']]
                        if i == 0:
                            est_data_gen_transformed_survae.append(out_df)
                        else:   
                            est_data_gen_transformed_survae[0] = pd.concat([est_data_gen_transformed_survae[0], out_df], ignore_index=True)
                        gen_shape = len(est_data_gen_transformed_survae[0])
                        i += 1

                    gen_data = SurvivalAnalysisDataLoader(est_data_gen_transformed_survae[0][:condition['n_samples']*n_generated_dataset], target_column=target_column, time_to_event_column=time_to_event_column)
                    scores = _evaluate_generation(gen_data, cond_full_data_loader, metrics_list)
                    
            elif method == 'train_train_gen_full':
                train_test_share = .8
                n_samples = data.shape[0]
                n_train_samples = int(train_test_share * n_samples)
                train_index = np.random.choice(n_samples, n_train_samples, replace=False)

                train_data = df.iloc[train_index]
                full_data_loader = SurvivalAnalysisDataLoader(df, target_column = "censor", time_to_event_column = "time")
                train_data_loader = SurvivalAnalysisDataLoader(train_data, target_column=target_column, time_to_event_column=time_to_event_column)
                model_survae_trial = model_survae(**params)

                # train on train data
                model_survae_trial.fit(train_data_loader)

                if condition is None:
                    n_gen_sample = n_generated_sample if n_generated_sample is not None else data.shape[0]
                    gen_data = model_survae_trial.generate(count=n_gen_sample*n_generated_dataset)
                    scores = _evaluate_generation(gen_data, full_data_loader, metrics_list)
                else:
                    raise NotImplementedError("Conditioning not implemented for method=train_train_gen_full")

            elif method == 'train_train_gen_test':
                train_test_share = .8
                n_samples = data.shape[0]
                n_train_samples = int(train_test_share * n_samples)
                train_index = np.random.choice(n_samples, n_train_samples, replace=False)
                test_index = [i for i in np.arange(n_samples) if i not in train_index]

                train_data, test_data = df.iloc[train_index], df.iloc[test_index]
                train_data_loader = SurvivalAnalysisDataLoader(train_data, target_column=target_column, time_to_event_column=time_to_event_column)
                test_data_loader = SurvivalAnalysisDataLoader(test_data, target_column=target_column, time_to_event_column=time_to_event_column)
                model_survae_trial = model_survae(**params) 

                # train on train data
                model_survae_trial.fit(train_data_loader)
                if condition is None:
                    n_gen_sample = n_generated_sample if n_generated_sample is not None else test_data.shape[0]
                    gen_data = model_survae_trial.generate(count=n_gen_sample*n_generated_dataset)
                    scores = _evaluate_generation(gen_data, test_data_loader, metrics_list)
                else:
                    raise NotImplementedError("Conditioning not implemented for method=train_train_gen_test")

            else:
                raise ValueError("Method not recognized. Choose among 'train_full_gen_full', 'train_train_gen_full', 'train_train_gen_test'")
            
            print(f"Scores: {scores}")
        # except Exception as e:  # invalid set of params
        #     print(f"{type(e).__name__}: {e}")
        #     print(params)
        #     raise optuna.TrialPruned()
        except optuna.TrialPruned:
            raise
        except Exception as e:  # invalid set of params
            print(f"{type(e).__name__}: {e}")
            print(params)
            if isinstance(e, ValueError) and "invalid values" in str(e).lower():
                raise optuna.exceptions.TrialPruned()
            raise
        return tuple(scores) if is_multi_objective else scores[0]

    
    db_file = study_name + '.db'
    if os.path.exists(db_file):
        print("This optuna study ({}) already exists. We load the study from the existing file.".format(db_file))
        study = optuna.load_study(study_name=study_name, storage='sqlite:///'+study_name+'.db')
    else: 
        create_kwargs = dict(
            study_name=study_name,
            storage=f'sqlite:///{db_file}',
        )
        if is_multi_objective:
            create_kwargs["directions"] = ["minimize"] * len(metrics_list)
            create_kwargs["sampler"] = optuna.samplers.NSGAIISampler(seed=seed)
        else:
            create_kwargs["direction"] = "minimize"
            create_kwargs["sampler"] = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(**create_kwargs)

        default_params = {'n_iter': 1000, 
                          'lr': 1e-3, 
                          'decoder_n_layers_hidden': 3, 
                          'weight_decay': 1e-5,
                          'batch_size': 200, 
                          'n_units_embedding': 500, 
                          'decoder_n_units_hidden': 500, 
                          'decoder_nonlin': 'leaky_relu', 
                          'decoder_dropout': 0, 
                          'encoder_n_layers_hidden': 3, 
                          'encoder_n_units_hidden': 500, 
                          'encoder_nonlin': 'leaky_relu',
                          'encoder_dropout': 0.1}
        study.enqueue_trial(default_params)
        print("Enqueued trial:", study.get_trials(deepcopy=False))
    study.optimize(objective, n_trials=n_trials)

    if is_multi_objective:
        best_trials = study.best_trials  # Pareto front
        pareto = [{"params": t.params, "values": t.values} for t in best_trials]
        best_to_return = pareto
        print(f"Pareto front: {len(best_trials)} trials")
    else:
        best_params = study.best_params
        best_to_return = best_params

    return best_to_return, study


def get_n_hyperparameters(generator_name):
    """
    Returns the number of hyperparameters for the SurVAE model.
    """
    model = type(Plugins().get("survae"))
    hp_space = model.hyperparameter_space()
    return len(hp_space)


def optuna_hyperparameter_search_val_loss(data, columns, target_column, time_to_event_column, n_trials, 
                                          study_name='optuna_study_survae', seed=10):
    
    df = pd.DataFrame(data.numpy(), columns=columns) # Preprocessed dataset
    data_loader = SurvivalAnalysisDataLoader(df, target_column=target_column, time_to_event_column=time_to_event_column)

    def objective(trial: optuna.Trial):
        set_seed()
        model_survae = type(Plugins().get("survae"))
        hp_space = model_survae.hyperparameter_space()
        # hp_space[0].high = 100  # speed up for now
        hp_space[3].choices = [1e-3, 1e-4, 1e-5]
        hp_space[4].choices = [64, 128, 200, 256, 512]
        params = suggest_all(trial, hp_space)
        ID = f"trial_{trial.number}"
        print(ID)
        score = None
        try:
            model_survae_trial = model_survae(**params)
            # train on full data
            model_survae_trial.fit(data_loader)
            score = model_survae_trial.best_val_loss
            print("Score:", score)
            
        except optuna.TrialPruned:
            raise
        except Exception as e:  # invalid set of params
            print(f"{type(e).__name__}: {e}")
            print(params)
            if isinstance(e, ValueError) and "invalid values" in str(e).lower():
                raise optuna.exceptions.TrialPruned()
            raise
        return score

    db_file = study_name + '.db'
    if os.path.exists(db_file):
        print("This optuna study ({}) already exists. We load the study from the existing file.".format(db_file))
        study = optuna.load_study(study_name=study_name, storage='sqlite:///'+study_name+'.db')
    else: 
        create_kwargs = dict(
            study_name=study_name,
            storage=f'sqlite:///{db_file}',
        )
        create_kwargs["direction"] = "minimize"
        create_kwargs["sampler"] = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(**create_kwargs)

        default_params = {'n_iter': 1000, 
                          'lr': 1e-3, 
                          'decoder_n_layers_hidden': 3, 
                          'weight_decay': 1e-5,
                          'batch_size': 200, 
                          'n_units_embedding': 500, 
                          'decoder_n_units_hidden': 500, 
                          'decoder_nonlin': 'leaky_relu', 
                          'decoder_dropout': 0, 
                          'encoder_n_layers_hidden': 3, 
                          'encoder_n_units_hidden': 500, 
                          'encoder_nonlin': 'leaky_relu',
                          'encoder_dropout': 0.1}
        study.enqueue_trial(default_params)
        print("Enqueued trial:", study.get_trials(deepcopy=False))
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_to_return = best_params

    return best_to_return, study