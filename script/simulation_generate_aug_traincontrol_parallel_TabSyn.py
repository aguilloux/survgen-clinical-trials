"""Persist naive TabSyn Monte Carlo datasets for the shared analysis script."""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter

from simulation_generate_aug_traincontrol_parallel import DEFAULT_DATASET, generate_one, make_config
from simulation_generate_aug_traincontrol_parallel_TabFORGE import positive_int, validate_requested_device

MODES = {'TabSyn-naive': {'mode': 'naive'}}


def run_tabsyn(generator, tensor, columns, feature_types, count, sample_size, params):
    import surv_tabsyn
    import torch
    import sklearn
    options = dict(params)
    mode = options.pop('mode')
    rounding = options.pop('apply_rounding')
    diagnostics = {'generation_retries': 0}
    started = perf_counter()
    generated = surv_tabsyn.run(
        tensor, columns, 'censor', 'time', count, n_generated_sample=sample_size,
        params=options, feat_types_dict=feature_types, apply_rounding=rounding,
        mode=mode, profile=diagnostics)
    diagnostics.update(total_seconds=perf_counter() - started,
                       versions={'torch': torch.__version__, 'sklearn': sklearn.__version__})
    diagnostics['seconds_per_synthetic_dataset'] = diagnostics['total_seconds'] / count
    print(f'{generator} timing: {json.dumps(diagnostics, sort_keys=True)}', flush=True)
    return generated, diagnostics


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--mc-start', type=positive_int, default=1)
    result.add_argument('--n-mc', type=positive_int, default=5)
    result.add_argument('--n-samples', type=positive_int, default=3000)
    result.add_argument('--n-generated-dataset', type=positive_int, default=50)
    result.add_argument('--control-fractions', nargs='+', type=float, default=[1/3])
    result.add_argument('--treatment-effects', nargs='+', type=float)
    result.add_argument('--generators', nargs='+', choices=list(MODES), default=list(MODES))
    result.add_argument('--output-dir', type=Path, default=DEFAULT_DATASET / 'generated_two_stage_TabSyn')
    result.add_argument('--tabsyn-root', type=Path, default=os.environ.get('TABSYN_ROOT'),
                        help='Official amazon-science/tabsyn checkout; defaults to TABSYN_ROOT.')
    result.add_argument('--device', default='auto')
    result.add_argument('--vae-epochs', type=positive_int, default=4000)
    result.add_argument('--diffusion-epochs', type=positive_int, default=10001)
    result.add_argument('--batch-size', type=positive_int, default=4096)
    result.add_argument('--sampling-steps', type=positive_int, default=50)
    result.add_argument('--denoiser-width', type=positive_int, default=1024,
                        help='Official width is 1024; smaller values are useful for smoke tests.')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if len(set(args.generators)) != len(args.generators):
        raise ValueError('Generator names must be unique.')
    if args.tabsyn_root is None:
        raise ValueError('Pass --tabsyn-root or set TABSYN_ROOT; see script/simulation_TabSyn.md.')
    if args.sampling_steps < 2 or args.denoiser_width < 4 or args.denoiser_width % 2:
        raise ValueError('Sampling steps must be >= 2; denoiser width must be even and >= 4.')
    validate_requested_device(args.device)
    config = make_config(args)
    from surv_tabsyn import source_fingerprint
    root = args.tabsyn_root.expanduser().resolve()
    shared = dict(tabsyn_root=str(root), source_sha256=source_fingerprint(root),
                  apply_rounding=True, random_state=1, device=args.device,
                  vae_epochs=args.vae_epochs, diffusion_epochs=args.diffusion_epochs,
                  batch_size=args.batch_size, sampling_steps=args.sampling_steps,
                  denoiser_width=args.denoiser_width)
    parameters = {name: dict(shared, **MODES[name]) for name in args.generators}
    output = args.output_dir.resolve()
    for d in range(len(config['control_fractions'])):
        for seed in range(args.mc_start, args.mc_start + args.n_mc):
            generate_one(config, seed, d, args.generators, parameters, output, runner=run_tabsyn)


if __name__ == '__main__':
    main()
