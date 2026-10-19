"""Generate TabPFN controls using sampled controls plus the full treated arm.

Each treatment effect has a separate model and archive root (effect_0, ...).
Pass an effect root to simulation_analyze_aug_traincontrol_parallel.py.
"""
from pathlib import Path

from simulation_generate_aug_traincontrol_parallel import (
    ROOT, decode, generate_one, make_config, simulation_kwargs,
)
from simulation_generate_aug_traincontrol_parallel_TabPFN import (
    TABPFN_MODES, parser as control_parser, run_tabpfn, validate_requested_device,
)

DEFAULT_OUTPUT = ROOT / 'dataset' / 'Simulations_aug_indep_trainfull' / 'generated_two_stage_TabPFN'


def trainfull_runner(config, seed, fraction_index):
    """Supply the shared archive writer with a full-arm conditional generator."""
    def run(generator, tensor, columns, feature_types, count, sample_size, params):
        import numpy as np
        import pandas as pd
        from simulations import simulation

        control, treated, types = simulation(
            config['treatment_effect_hyperopt'], seed=seed, **simulation_kwargs(config))
        control = control.iloc[:int(config['control_fractions'][fraction_index] * len(control))]
        if control.empty or treated.empty:
            raise ValueError('Trainfull generation requires nonempty control and treated arms.')
        # simulation() places treatment before the survival pair, but omits its
        # descriptor. Include it as a categorical predictor for both model stages.
        covariates = list(control.drop(columns=['treatment', 'time', 'censor']).columns)
        training = pd.concat([control, treated], ignore_index=True)
        training = training[covariates + ['treatment', 'time', 'censor']]
        training['treatment'] = training['treatment'].astype(np.float32)
        treatment_type = pd.DataFrame([
            {'name': 'treatment', 'type': 'cat', 'dim': 1, 'nclass': '2'}])
        training_types = pd.concat([types.iloc[:-1], treatment_type, types.iloc[-1:]], ignore_index=True)
        # Use inferred types, including the categorical treatment descriptor.
        _, inferred, _, _, training_tensor, training_frame = decode(
            training, training_types, Path.cwd())
        generated, diagnostics = run_tabpfn(
            generator, training_tensor, list(training_frame.columns), inferred,
            count, sample_size, params)
        treatment_index = training_frame.columns.get_loc('treatment')
        keep = [training_frame.columns.get_loc(name) for name in columns]
        for data in generated:
            if not bool((data[:, treatment_index] == 0).all()):
                raise ValueError('Conditional generator returned non-control rows.')
        diagnostics.update(training_controls=len(control), training_treated=len(treated))
        # Shared analysis expects treatment to be implicit in synthetic controls.
        return [data[:, keep] for data in generated], diagnostics
    return run


def parser():
    result = control_parser()
    result.description = __doc__
    result.set_defaults(output_dir=DEFAULT_OUTPUT)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if len(set(args.generators)) != len(args.generators):
        raise ValueError('Generator names must be unique.')
    validate_requested_device(args.device)
    base_config = make_config(args)
    parameters = {}
    for generator in args.generators:
        params = dict(t=1.0, n_permutations=1, apply_rounding=True,
                      cond_gen={'var': 'treatment', 'value': 0}, **TABPFN_MODES[generator])
        for key in ('device', 'fit_mode', 'n_estimators'):
            value = getattr(args, key)
            if value is not None:
                params[key] = value
        parameters[generator] = params
    for index, effect in enumerate(base_config['treatment_effects']):
        config = dict(base_config, training_mode='trainfull',
                      treatment_effect_hyperopt=effect, treatment_effects=[effect])
        output = args.output_dir.resolve() / f'effect_{index}'
        for d in range(len(config['control_fractions'])):
            for seed in range(args.mc_start, args.mc_start + args.n_mc):
                generate_one(config, seed, d, args.generators, parameters, output,
                             runner=trainfull_runner(config, seed, d))


if __name__ == '__main__':
    main()
