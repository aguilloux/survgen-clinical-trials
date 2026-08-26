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

def run_worker(return_dict, model, params, data, count, cond, n_generated_dataset, cond_gen):
    # print("training....")
    model_trial = model(**params)
    model_trial.fit(data, cond=cond)
    # print("generation....")
    if cond_gen is None:
        cond_gen = pd.concat([cond for i in range(n_generated_dataset)]) 
    else:
        cond_gen = pd.concat([cond_gen for i in range(n_generated_dataset)])    
    result = model_trial.generate(count=count, cond=cond_gen)
    return_dict["result"] = result

def run_with_timeout_mp(model, params, data, count, cond, n_generated_dataset, cond_gen, timeout=60):
    manager = mp.Manager()
    return_dict = manager.dict()
    p = mp.Process(target=run_worker, args=(return_dict, model, params, data, count, cond, n_generated_dataset, cond_gen))
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

def run(data, columns, target_column, time_to_event_column, n_generated_dataset, n_generated_sample=None, params=None, cond_gen=None, apply_rounding=False, feat_types_dict=None):
    """
    Use a conditional GAN for survival data generation
    """

    set_seed()
    
    # Define data and model
    df = pd.DataFrame(data.numpy(), columns=columns) # Preprocessed dataset
    data = SurvivalAnalysisDataLoader(df, target_column=target_column, time_to_event_column=time_to_event_column)
    
    if params is not None:
        model = type(Plugins().get("survival_gan"))
        model_survgan = model(**params)
    else:
        model_survgan = Plugins().get("survival_gan") 
        print(model_survgan.__dict__)

    # Train
    if cond_gen is not None:
        cond = df[cond_gen.columns]
    else:
        cond = df[[target_column]]
    model_survgan.fit(data, cond=cond)
    
    # Generate
    if isinstance(n_generated_sample, list):
        est_data_gen_transformed_survgan_list = []
        for n_generated_sample_ in n_generated_sample:
            if cond_gen is None:
                indices = torch.cat((torch.arange(0, data.shape[0]), torch.randint(0, data.shape[0], (n_generated_sample_ - data.shape[0],))))
                cond_gen = SurvivalAnalysisDataLoader(df.loc[indices], target_column=target_column, time_to_event_column=time_to_event_column)[[target_column]]
            est_data_gen_transformed_survgan = []
            for j in range(n_generated_dataset):
                out = model_survgan.generate(count=n_generated_sample_, cond=cond_gen)
                if apply_rounding:
                    out = data_processing.round_data_gen(df.values, torch.from_numpy(out.numpy()), feat_types_dict)
                est_data_gen_transformed_survgan.append(out)

            est_data_gen_transformed_survgan_list.append(est_data_gen_transformed_survgan)

        return est_data_gen_transformed_survgan_list
    else:
        if cond_gen is None:
            if n_generated_sample is None:
                n_generated_sample = data.shape[0]
            indices = torch.cat((torch.arange(0, data.shape[0]), torch.randint(0, data.shape[0], (n_generated_sample - data.shape[0],))))
            cond_gen = SurvivalAnalysisDataLoader(df.loc[indices], target_column=target_column, time_to_event_column=time_to_event_column)[[target_column]]
        else:
            n_generated_sample = cond_gen.shape[0]
    
        est_data_gen_transformed_survgan = []
        for j in range(n_generated_dataset):
            out = model_survgan.generate(count=n_generated_sample, cond=cond_gen)
            if apply_rounding:
                out = data_processing.round_data_gen(df.values, torch.from_numpy(out.numpy()), feat_types_dict)
            est_data_gen_transformed_survgan.append(out)

        return est_data_gen_transformed_survgan



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
                                 n_generated_sample=None, cond_gen=None, study_name='optuna_study_surv_gan', 
                                 metric='survival_km_distance', method='', cond_df=None, seed=10, apply_rounding=False, feat_types_dict=None):
    
    df = pd.DataFrame(data.numpy(), columns=columns) # Preprocessed dataset
    dataloader = SurvivalAnalysisDataLoader(df, target_column=target_column, time_to_event_column=time_to_event_column)
    if cond_gen is not None:
        cond_generation = cond_gen.copy()
        cond_dataloader = SurvivalAnalysisDataLoader(cond_df, target_column=target_column, time_to_event_column=time_to_event_column)
    else:
        cond_generation = None
        cond_dataloader = None

    metrics_list = metric if isinstance(metric, list) else [metric]
    print(f"Metrics for optimization: {metrics_list}")
    is_multi_objective = len(metrics_list) > 1

    def objective(trial: optuna.Trial):
        set_seed(seed=seed)
        model = type(Plugins().get("survival_gan"))
        hp_space = model.hyperparameter_space()
        hp_space[0].high = 3  # speed up for now
        params = suggest_all(trial, hp_space)
        ID = f"trial_{trial.number}"
        print(ID)
        scores = []
        try:
            if method == 'train_full_gen_full':
                if cond_generation is None:
                    n_gen_sample = n_generated_sample if n_generated_sample is not None else data.shape[0]
                    indices = torch.cat((torch.arange(0, data.shape[0]), torch.randint(0, data.shape[0], (max(0, n_gen_sample - data.shape[0]),))))
                    # cond = SurvivalAnalysisDataLoader(df.loc[indices], target_column=target_column, time_to_event_column=time_to_event_column)[[target_column]]
                    cond_gen = df.loc[indices][[target_column]]
                    cond = df[[target_column]]
                    gen_data = run_with_timeout_mp(model, params, dataloader, n_gen_sample*n_generated_dataset, cond, n_generated_dataset, cond_gen, timeout=120)
                    if apply_rounding:
                        out_rounded = data_processing.round_data_gen(df.values, torch.from_numpy(gen_data.numpy()), feat_types_dict)
                        out_df = pd.DataFrame(out_rounded, columns=columns) # Preprocessed dataset
                        gen_data = SurvivalAnalysisDataLoader(out_df, target_column=target_column, time_to_event_column=time_to_event_column)
                    scores = _evaluate_generation(gen_data, dataloader, metrics_list)
                else:
                    cond = df[cond_generation.columns]
                    gen_data = run_with_timeout_mp(model, params, dataloader, cond_generation.shape[0]*n_generated_dataset, cond, n_generated_dataset, cond_generation, timeout=120)
                    scores = _evaluate_generation(gen_data, cond_dataloader, metrics_list)
            
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

        default_params = {'generator_n_layers_hidden': 2, 
                          'generator_n_units_hidden': 500, 
                          'generator_nonlin': 'relu', 
                          'generator_dropout': 0.1, 
                          'discriminator_n_layers_hidden': 2, 
                          'discriminator_n_units_hidden': 500, 
                          'discriminator_nonlin': 'leaky_relu', 
                          'discriminator_dropout': 0.1, 
                          'lr':  1e-3, 
                          'weight_decay': 1e-3, 
                          'encoder_max_clusters': 5}
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
    Returns the number of hyperparameters for the SurvGAN model.
    """
    model = type(Plugins().get("survival_gan"))
    hp_space = model.hyperparameter_space()
    return len(hp_space)


def optuna_hyperparameter_search_val_loss(data, columns, target_column, time_to_event_column, n_trials, 
                                        study_name='optuna_study_surv_gan', seed=10):
    
    df = pd.DataFrame(data.numpy(), columns=columns) # Preprocessed dataset
    dataloader = SurvivalAnalysisDataLoader(df, target_column=target_column, time_to_event_column=time_to_event_column)
    cond = df[[target_column]]

    def objective(trial: optuna.Trial):
        set_seed(seed=seed)
        model = type(Plugins().get("survival_gan"))
        hp_space = model.hyperparameter_space()
        hp_space[0].high = 3  # speed up for now
        params = suggest_all(trial, hp_space)
        ID = f"trial_{trial.number}"
        print(ID)
        score = None
        try:
            model_survgan_trial = model(**params)
            model_survgan_trial.fit(dataloader, cond=cond)
            score = model_survgan_trial.best_val_loss
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

        default_params = {'generator_n_layers_hidden': 2, 
                          'generator_n_units_hidden': 500, 
                          'generator_nonlin': 'relu', 
                          'generator_dropout': 0.1, 
                          'discriminator_n_layers_hidden': 2, 
                          'discriminator_n_units_hidden': 500, 
                          'discriminator_nonlin': 'leaky_relu', 
                          'discriminator_dropout': 0.1, 
                          'lr':  1e-3, 
                          'weight_decay': 1e-3, 
                          'encoder_max_clusters': 5}
        study.enqueue_trial(default_params)
        print("Enqueued trial:", study.get_trials(deepcopy=False))
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    best_to_return = best_params

    return best_to_return, study