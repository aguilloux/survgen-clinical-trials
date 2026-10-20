"""Train generators and persist Monte Carlo datasets; see simulation_two_stage.md."""
import argparse
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / 'utils', ROOT / 'execute'):
    sys.path.insert(0, str(directory))

DEFAULT_GENERATORS = [
    'HI-VAE_weibull', 'HI-VAE_piecewise', 'Surv-GAN', 'Surv-VAE',
    'HI-VAE_weibull_prior', 'HI-VAE_piecewise_prior',
]
GENERATORS = DEFAULT_GENERATORS + [
    'HI-VAE_lognormal', 'HI-VAE_weibull_DP', 'HI-VAE_piecewise_DP',
    'HI-VAE_weibull_diffusion', 'HI-VAE_piecewise_diffusion',
    'HI-VAE_weibull_diffusion_DP', 'HI-VAE_piecewise_diffusion_DP',
]
DEFAULT_DATASET = ROOT / 'dataset' / 'Simulations_aug_indep_traincontrol'


@contextmanager
def private_workspace():
    """Isolate generator files and Synthcity caches, including on failure."""
    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix='survgen_', dir=os.environ.get('SCRATCH') or None) as work:
        try:
            os.chdir(work)
            yield Path(work).resolve()
        finally:
            os.chdir(previous)


def atomic_npz(path, **arrays):
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def rng_arrays(prefix='rng'):
    import numpy as np
    algorithm, keys, pos, has_gauss, cached = np.random.get_state()
    return {prefix + '_keys': keys, prefix + '_pos': pos,
            prefix + '_has_gauss': has_gauss, prefix + '_cached': cached}


def restore_rng(archive, prefix='rng'):
    import numpy as np
    np.random.set_state(('MT19937', archive[prefix + '_keys'],
                         int(archive[prefix + '_pos']), int(archive[prefix + '_has_gauss']),
                         float(archive[prefix + '_cached'])))


def adjust_feat_types_for_generator(generator_name, feat_types_dict):
    result = [dict(ft) for ft in feat_types_dict]
    for ft in result:
        if ft['name'] == 'survcens':
            ft['type'] = ('surv_weibull' if 'HI-VAE_weibull' in generator_name else
                          'surv' if generator_name == 'HI-VAE_lognormal' else 'surv_piecewise')
    return result


def simulation_kwargs(config):
    return {key: config[key] for key in (
        'n_samples', 'independent', 'feature_types_list', 'n_features_bytype',
        'n_active_features', 'p_treated', 'shape_T', 'shape_C', 'scale_C',
        'scale_C_indep', 'data_types_create')}


def decode(raw, types, work, feature_types=None):
    """Keep the original CSV/read_data preprocessing (including category encoding)."""
    import torch
    import pandas as pd
    import data_processing
    raw.to_csv(work / 'data.csv', index=False, header=False)
    types.to_csv(work / 'types.csv', index=False)
    encoded, inferred, miss, true_miss, _ = data_processing.read_data(
        str(work / 'data.csv'), str(work / 'types.csv'), miss_file='Missing.csv', true_miss_file=None)
    tensor = data_processing.discrete_variables_transformation(
        torch.from_numpy(encoded.values), inferred if feature_types is None else feature_types)
    columns = types['name'][:-1].tolist() + ['time', 'censor']
    return encoded, inferred, miss, true_miss, tensor, pd.DataFrame(tensor.numpy(), columns=columns)


def metadata(archive):
    return json.loads(str(archive['metadata']))


class ConfigurationMismatch(ValueError):
    """An existing archive belongs to another experiment configuration."""


def archive_complete(path, expected, required=(), expected_data_shape=None):
    """Return whether an existing archive is complete enough to be resumed.

    Configuration mismatches remain hard errors because replacing an archive
    from another experiment would silently mix results. Corrupt or incomplete
    archives are safe to regenerate because ``atomic_npz`` replaces them only
    after the new archive has been written successfully.
    """
    import numpy as np
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            if metadata(archive) != expected:
                raise ConfigurationMismatch(
                    f'Configuration mismatch in {path}; use a separate --output-dir.')
            missing = [name for name in required if name not in archive.files]
            if missing:
                raise KeyError(f'missing arrays {missing}')
            if expected_data_shape is not None and archive['data'].shape != expected_data_shape:
                raise ValueError(
                    f"data has shape {archive['data'].shape}, expected {expected_data_shape}")
    except ConfigurationMismatch:
        raise
    except Exception as error:
        print(f'Regenerating incomplete archive {path}: {error}', flush=True)
        return False
    return True


def reference_requirements(n_treatment_effects):
    required = [
        'columns', 'control', 'control_dtypes', 'raw_control_dtypes',
        'raw_columns', 'raw_control', 'feature_types',
    ]
    for index in range(n_treatment_effects):
        required.extend([
            f'treated_{index}', f'treated_{index}_dtypes',
            f'raw_treated_{index}', f'raw_treated_{index}_dtypes',
        ])
    return required


def generated_shape_from_reference(path, count):
    """Read the exact expected generator output shape from a reference archive."""
    import numpy as np
    with np.load(path, allow_pickle=False) as archive:
        sample_size = max(
            len(archive['raw_control']),
            *(len(archive[f'raw_treated_{index}'])
              for index in range(len(metadata(archive)['config']['treatment_effects'])))
        )
        return count, sample_size, len(archive['columns']) - 1


def generated_to_numpy(dataset, columns):
    """Convert supported generator outputs to a numeric array in canonical order.

    HI-VAE and TabPFN return tensors, while Synthcity's Surv-GAN and Surv-VAE
    return SurvivalAnalysisDataLoader objects.  Avoid importing Synthcity here:
    its loaders expose ``dataframe()``, which also lets us enforce column order.
    """
    import numpy as np
    import pandas as pd
    import torch

    if torch.is_tensor(dataset):
        return dataset.detach().cpu().numpy()
    if isinstance(dataset, pd.DataFrame):
        frame = dataset
    elif callable(getattr(dataset, 'dataframe', None)):
        frame = dataset.dataframe()
    else:
        converter = getattr(dataset, 'numpy', None)
        if callable(converter):
            return np.asarray(converter())
        raise TypeError(
            f'Unsupported generated dataset type {type(dataset).__module__}.'
            f'{type(dataset).__qualname__}; expected a tensor, DataFrame, '
            'Synthcity dataloader, or object exposing numpy().'
        )

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f'Generated dataset is missing columns: {missing}')
    return frame.loc[:, columns].to_numpy()


def generate_one(config, seed, fraction_index, generators, best_params, output, runner=None):
    """Persist one context; an optional runner supplies another generator family.

    runner(name, tensor, columns, feature_types, count, sample_size, params)
    returns (list_of_tensors, JSON_serializable_diagnostics).
    """
    import numpy as np
    from simulations import simulation
    fraction = config['control_fractions'][fraction_index]
    destination = output / f'MC_{seed:06d}' / f'control_{fraction_index}'
    reference = destination / 'reference.npz'
    meta = {'schema_version': 1, 'config': config, 'mc_seed': seed,
            'fraction_index': fraction_index}
    reference_complete = archive_complete(
        reference, meta, reference_requirements(len(config['treatment_effects'])))
    pending = None
    if reference_complete:
        expected_shape = generated_shape_from_reference(
            reference, config['n_generated_dataset'])
        pending = [
            generator for generator in generators
            if not archive_complete(
                destination / (generator + '.npz'),
                dict(meta, generator=generator, params=best_params[generator]),
                required=('data',), expected_data_shape=expected_shape)
        ]
    if reference_complete and not pending:
        print(f'Skipping complete MC {seed}, control fraction {fraction}', flush=True)
        return
    with private_workspace() as work:
        control, treated, types = simulation(config['treatment_effect_hyperopt'], seed=seed,
                                             **simulation_kwargs(config))
        control = control.iloc[:int(fraction * len(control))].drop(columns='treatment')
        treated = treated.drop(columns='treatment')
        encoded, feature_types, miss, true_miss, tensor, decoded = decode(control, types, work)
        decoded['treatment'] = 0
        # Build and persist reference datasets before training. Model wrappers reset their
        # own RNGs; restore the simulation state as well to preserve their input state.
        simulation_rng = rng_arrays()
        if not reference_complete:
            arrays = dict(metadata=json.dumps(meta), columns=np.asarray(decoded.columns, dtype=str),
                          control=decoded.to_numpy(),
                          control_dtypes=np.asarray(decoded.dtypes.astype(str), dtype=str),
                          raw_control_dtypes=np.asarray(control.dtypes.astype(str), dtype=str),
                          raw_columns=np.asarray(control.columns, dtype=str),
                          raw_control=control.to_numpy(), feature_types=json.dumps(feature_types))
            for t, effect in enumerate(config['treatment_effects']):
                _, raw_treated, treated_types = simulation(effect, seed=seed, **simulation_kwargs(config))
                raw_treated = raw_treated.drop(columns='treatment')
                arrays.update(rng_arrays(f'treated_rng_{t}'))
                *_, decoded_treated = decode(raw_treated, treated_types, work, feature_types)
                decoded_treated['treatment'] = 1
                arrays[f'treated_{t}'] = decoded_treated.to_numpy()
                arrays[f'treated_{t}_dtypes'] = np.asarray(decoded_treated.dtypes.astype(str), dtype=str)
                arrays[f'raw_treated_{t}_dtypes'] = np.asarray(raw_treated.dtypes.astype(str), dtype=str)
                arrays[f'raw_treated_{t}'] = raw_treated.to_numpy()
            atomic_npz(reference, **arrays)
        restore_rng(simulation_rng)
        expected_shape = (config['n_generated_dataset'], max(len(treated), len(control)), len(control.columns))
        if pending is None:
            pending = [
                generator for generator in generators
                if not archive_complete(
                    destination / (generator + '.npz'),
                    dict(meta, generator=generator, params=best_params[generator]),
                    required=('data',), expected_data_shape=expected_shape)
            ]
        for generator in pending:
            print(f'Generating MC {seed}, control fraction {fraction}, {generator}', flush=True)
            params = dict(best_params[generator])
            diagnostics = {}
            if runner is not None:
                generated, diagnostics = runner(
                    generator, tensor, list(decoded.columns[:-1]), feature_types,
                    config['n_generated_dataset'], max(len(treated), len(control)), params)
            elif 'HI-VAE' in generator:
                module = importlib.import_module('surv_hivae')
                generated = module.run(
                    encoded, miss, true_miss, adjust_feat_types_for_generator(generator, feature_types),
                    config['n_generated_dataset'], params=params,
                    gen_from_prior='_prior' in generator, norm_mode='global',
                    n_generated_sample=max(len(treated), len(control)),
                    differential_privacy='_DP' in generator, diffusion='_diffusion' in generator,
                    diffusion_var='z_and_s', generation_level='encoder')
            else:
                module = importlib.import_module({'Surv-GAN': 'surv_gan', 'Surv-VAE': 'surv_vae'}[generator])
                generated = module.run(
                    tensor, columns=list(decoded.columns[:-1]), target_column='censor',
                    time_to_event_column='time', n_generated_dataset=config['n_generated_dataset'],
                    n_generated_sample=max(len(treated), len(control)), params=params)
            generated_columns = list(decoded.columns[:-1])
            values = np.stack([
                generated_to_numpy(data, generated_columns) for data in generated
            ])
            if values.shape != expected_shape:
                raise ValueError(f'{generator}: expected {expected_shape}, received {values.shape}')
            atomic_npz(destination / (generator + '.npz'), data=values,
                       metadata=json.dumps(dict(meta, generator=generator, params=params)),
                       diagnostics=json.dumps(diagnostics), **rng_arrays())
            del generated, values


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--mc-start', type=int, default=1, help='First one-based MC seed (inclusive).')
    result.add_argument('--n-mc', type=int, default=50, help='Number of independent real-data replicates.')
    result.add_argument('--n-generated-dataset', type=int, default=200)
    result.add_argument('--n-samples', type=int, default=600)
    result.add_argument('--control-fractions', nargs='+', type=float, default=[1/3],
                        help='Fractions of the simulated control group used for training.')
    result.add_argument('--generators', nargs='+', choices=GENERATORS, default=DEFAULT_GENERATORS)
    result.add_argument('--treatment-effects', nargs='+', type=float, default=None)
    result.add_argument('--params-dir', type=Path, default=DEFAULT_DATASET / 'optuna_results')
    result.add_argument('--output-dir', type=Path, default=DEFAULT_DATASET / 'generated_two_stage')
    return result


def make_config(args):
    """Shared experiment definition for neural generators and TabPFN."""
    import numpy as np
    config = dict(
        n_samples=args.n_samples,
        n_features_bytype=6,
        n_active_features=3,
        p_treated=0.5,
        shape_T=2.,
        shape_C=2.,
        scale_C=2.5,
        scale_C_indep=3.9,
        feature_types_list=['real', 'cat'],
        independent=True,
        data_types_create=True,
        treatment_effect_hyperopt=0.,
        metric_optuna='detection_xgb',
        n_trials=150,
        control_fractions=args.control_fractions,
        treatment_effects=args.treatment_effects if args.treatment_effects is not None else np.arange(0., 1.1, 0.2).tolist(),
        n_generated_dataset=args.n_generated_dataset)

    if not all(np.isfinite(config['treatment_effects'])):
        raise ValueError('Treatment effects must be finite.')
    if not all(0 < fraction <= 1 for fraction in config['control_fractions']):
        raise ValueError('Control fractions must be in (0, 1].')
    return config


def main(argv=None):
    args = parser().parse_args(argv)
    if min(args.mc_start, args.n_mc, args.n_generated_dataset, args.n_samples) < 1:
        raise ValueError('MC start, counts and sample size must be positive.')
    if len(set(args.generators)) != len(args.generators):
        raise ValueError('Generator names must be unique.')

    config = make_config(args)
    output = args.output_dir.resolve()
    # Preflight every parameter file before any expensive training.
    parameters = []
    for fraction in config['control_fractions']:
        # name = f'simu_N{args.n_samples}_Ncontrol{int(fraction * 3 + 0.01)}%3_nfeat6_t0'
        name = f'simu_N600_Ncontrol{int(fraction * 3 + 0.01)}%3_nfeat6_t0'
        by_generator = {}
        for generator in args.generators:
            path = args.params_dir / f'best_params_{name}_ntrials150_detection_xgb_{generator}.json'
            by_generator[generator] = json.loads(path.read_text())
            if 'HI-VAE' in generator:
                by_generator[generator]['epochs'] = 5000
        parameters.append(by_generator)
    for d, best_params in enumerate(parameters):
        for seed in range(args.mc_start, args.mc_start + args.n_mc):
            generate_one(config, seed, d, args.generators, best_params, output)


if __name__ == '__main__':
    main()
