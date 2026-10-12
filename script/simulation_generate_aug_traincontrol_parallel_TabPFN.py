"""Persist independent TabPFN Monte Carlo contexts for the shared analysis script.

Defaults reproduce simulation_experiment_aug_traincontrol_parallel_TabPFN.py.
See simulation_two_stage.md for timing, batching limitations, and Slurm examples.
"""
import argparse
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
from time import perf_counter

from simulation_generate_aug_traincontrol_parallel import DEFAULT_DATASET, generate_one, make_config

TABPFN_MODES = {
    'TabPFN-naive': {'mode': 'naive'},
    'TabPFN-survival_function-joint': {
        'mode': 'survival_function', 'censoring_strategy': 'joint',
        'tte_model': 'survival_function_regression',
    },
}


def run_tabpfn(generator, tensor, columns, feature_types, count, sample_size, params):
    # Import only for incomplete contexts; --help and completed resumes need no TabPFN.
    import surv_tabpfn
    import torch
    options = dict(params)
    mode = options.pop('mode')
    rounding = options.pop('apply_rounding')
    diagnostics = {'generation_retries': 0}
    started = perf_counter()
    generated = surv_tabpfn.run(
        tensor, 
        columns=columns, 
        target_column='censor', 
        time_to_event_column='time',
        n_generated_dataset=count, 
        n_generated_sample=sample_size,
        params=options, feat_types_dict=feature_types, apply_rounding=rounding,
        mode=mode, profile=diagnostics)
    diagnostics['total_seconds'] = perf_counter() - started
    diagnostics['seconds_per_synthetic_dataset'] = diagnostics['total_seconds'] / count
    diagnostics['versions'] = {}
    for package in ('tabpfn', 'tabpfn-extensions', 'torch', 'synthcity'):
        try:
            diagnostics['versions'][package] = version(package)
        except PackageNotFoundError:
            diagnostics['versions'][package] = 'not installed'
    diagnostics['requested_device'] = options.get('device', 'auto')
    diagnostics['cuda_available'] = torch.cuda.is_available()
    print(f'{generator} timing: {json.dumps(diagnostics, sort_keys=True)}', flush=True)
    return generated, diagnostics


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError('Must be positive.')
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--mc-start', type=positive_int, default=1)
    result.add_argument('--n-mc', type=positive_int, default=5)
    result.add_argument('--n-samples', type=positive_int, default=1200)
    result.add_argument('--n-generated-dataset', type=positive_int, default=200)
    result.add_argument('--treatment-effects', nargs='+', type=float)
    result.add_argument('--generators', nargs='+', choices=list(TABPFN_MODES), default=list(TABPFN_MODES))
    result.add_argument('--output-dir', type=Path, default=DEFAULT_DATASET / 'generated_two_stage')
    result.add_argument('--device', default=None, help='TabPFN device, e.g. cuda or cpu. Omitted: legacy auto.')
    result.add_argument('--fit-mode', choices=['fit_preprocessors', 'fit_with_cache', 'low_memory'],
                        help='Optional inference strategy to benchmark; omitted: installed default.')
    result.add_argument('--n-estimators', type=positive_int,
                        help='Optional ensemble size. Changing it changes the generator; use a separate output directory.')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if len(set(args.generators)) != len(args.generators):
        raise ValueError('Generator names must be unique.')
    config = make_config(args)
    parameters = {}
    for generator in args.generators:
        params = dict(t=1.0, n_permutations=1, apply_rounding=True, **TABPFN_MODES[generator])
        for key in ('device', 'fit_mode', 'n_estimators'):
            value = getattr(args, key)
            if value is not None:
                params[key] = value
        parameters[generator] = params
    output = args.output_dir.resolve()
    for d in range(len(config['control_fractions'])):
        for seed in range(args.mc_start, args.mc_start + args.n_mc):
            generate_one(config, seed, d, args.generators, parameters, output, runner=run_tabpfn)


if __name__ == '__main__':
    main()
