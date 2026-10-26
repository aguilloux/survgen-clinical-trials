"""Persist naive TabFORGE Monte Carlo datasets for the shared analysis script."""
import argparse
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
from time import perf_counter
from simulation_generate_aug_traincontrol_parallel import DEFAULT_DATASET, generate_one, make_config

MODES = {
    'TabFORGE-naive': {'mode': 'naive'},
}


def run_tabforge(generator, tensor, columns, feature_types, count, sample_size, params):
    # Import only for incomplete contexts; --help and completed resumes need no TabFORGE.
    import surv_tabforge
    import torch
    options = dict(params)
    mode = options.pop('mode')
    rounding = options.pop('apply_rounding')
    condition = options.pop('cond_gen', None)
    diagnostics = {'generation_retries': 0}
    started = perf_counter()
    generated = surv_tabforge.run(
        tensor, 
        columns=columns, 
        target_column='censor', 
        time_to_event_column='time',
        n_generated_dataset=count, 
        n_generated_sample=sample_size,
        params=options, feat_types_dict=feature_types, apply_rounding=rounding,
        mode=mode, profile=diagnostics, cond_gen=condition)
    diagnostics['total_seconds'] = perf_counter() - started
    diagnostics['seconds_per_synthetic_dataset'] = diagnostics['total_seconds'] / count
    diagnostics['versions'] = {}
    for package in ('tabforge-ml', 'torch', 'synthcity'):
        try:
            diagnostics['versions'][package] = version(package)
        except PackageNotFoundError:
            diagnostics['versions'][package] = 'not installed'
    diagnostics['requested_device'] = options.get('runtime_config', {}).get('device', 'auto')
    diagnostics['cuda_available'] = torch.cuda.is_available()
    print(f'{generator} timing: {json.dumps(diagnostics, sort_keys=True)}', flush=True)
    return generated, diagnostics


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError('Must be positive.')
    return result


def validate_requested_device(device):
    """Fail before simulation when an explicitly requested CUDA device is absent."""
    if device is None or not device.lower().startswith('cuda'):
        return
    import torch
    if torch.cuda.is_available():
        return
    allocation = {
        name: os.environ.get(name, '<unset>')
        for name in ('SLURM_JOB_ID', 'SLURM_JOB_GPUS', 'CUDA_VISIBLE_DEVICES')
    }
    raise RuntimeError(
        f'--device {device} was requested, but torch.cuda.is_available() is False. '
        f'Slurm/CUDA environment: {allocation}. Ensure the batch directive is '
        "exactly '#SBATCH --gres=gpu:1' (no space between # and SBATCH), "
        'then verify nvidia-smi and the CUDA-enabled PyTorch environment.'
    )


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--mc-start', type=positive_int, default=1)
    result.add_argument('--n-mc', type=positive_int, default=5)
    result.add_argument('--n-samples', type=positive_int, default=1200)
    result.add_argument('--n-generated-dataset', type=positive_int, default=200)
    result.add_argument('--control-fractions', nargs='+', type=float, default=[1/3],
                        help='Fractions of the simulated control group used for generation.')
    result.add_argument('--treatment-effects', nargs='+', type=float)
    result.add_argument('--generators', nargs='+', choices=list(MODES), default=list(MODES))
    result.add_argument('--output-dir', type=Path, default=DEFAULT_DATASET / 'generated_two_stage_TabFORGE')
    result.add_argument('--device', default='auto', help='TabFORGE device: cuda, cpu, or auto.')
    result.add_argument('--max-steps', type=positive_int, default=10000)
    result.add_argument('--batch-size', type=positive_int, default=512)
    result.add_argument('--checkpoint', default='pretrained',
                        help='Pretrained backbone or local checkpoint path; "none" trains from scratch.')
    result.add_argument('--embedding-model-path', default='auto',
                        help='Encoder checkpoint path, auto, or mock (smoke tests only).')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if len(set(args.generators)) != len(args.generators):
        raise ValueError('Generator names must be unique.')
    validate_requested_device(args.device)
    config = make_config(args)
    parameters = {}
    for generator in args.generators:
        params = dict(
            apply_rounding=True, **MODES[generator],
            runtime_config={'device': args.device},
            training_config={'max_steps': args.max_steps, 'batch_size': args.batch_size},
            embedding_config={'model_path': args.embedding_model_path},
            checkpoint=None if args.checkpoint.lower() == 'none' else args.checkpoint)
        parameters[generator] = params
    output = args.output_dir.resolve()
    for d in range(len(config['control_fractions'])):
        for seed in range(args.mc_start, args.mc_start + args.n_mc):
            generate_one(config, seed, d, args.generators, parameters, output, runner=run_tabforge)


if __name__ == '__main__':
    main()
