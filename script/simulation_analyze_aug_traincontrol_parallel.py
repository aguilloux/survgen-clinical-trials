"""Analyze saved Monte Carlo datasets without importing or training generators."""
import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import tempfile

from simulation_generate_aug_traincontrol_parallel import (
    DEFAULT_DATASET, GENERATORS, metadata, private_workspace,
    restore_rng, simulation_kwargs,
)

from simulation_generate_aug_traincontrol_parallel_TabPFN import TABPFN_MODES

ANALYSIS_GENERATORS = GENERATORS + list(TABPFN_MODES)

METRICS = ['J-S distance', 'KS test', 'Survival curves distance', 'Detection XGB',
           'NNDR', 'K-map score', 'Identifiability score', 'TableOne min p-value']


@lru_cache(maxsize=128)
def reference_coefficient(effect, seed, config_json):
    """Legacy 100,000-subject reference fit, cached only for identical inputs.

    Deliberately retains the original simulation defaults: p_treated=0.5 and
    data_types_create=True. The legacy helper did not forward these arguments.
    """
    import pandas as pd
    from simulations import simulation
    from metrics import fit_cox_model
    kwargs = simulation_kwargs(json.loads(config_json))
    kwargs.update(n_samples=100000, p_treated=0.5, data_types_create=True)
    control, treated, _ = simulation(effect, seed=seed, **kwargs)
    return fit_cox_model(pd.concat([control, treated], ignore_index=True),
                         ['time', 'censor', 'treatment'])[0][0]


def load_frame(archive, key, columns):
    import pandas as pd
    return pd.DataFrame(archive[key], columns=columns).astype(
        dict(zip(columns, archive[key + '_dtypes'].tolist())))


def procova_baselines(control, treated, p_treated, prognostic_score=None,
                      allow_cox_convergence_failure=False):
    """Compute the notebook's covariate-adjusted tests on the real trial arms.

    All returned values use the analysis CSV's negative-log-p-value convention.
    The prognostic score is cross-fitted once and shared by its log-rank and Cox
    tests; their score construction is otherwise identical in ``metrics.py``.
    """
    import numpy as np
    import pandas as pd
    import warnings
    from metrics import (
        compute_adjusted_logrank_test, compute_martingale_score,
        cox_adjusted_pvalue, martingale_score_adjusted_pvalue,
        prognostic_adjusted_logrank,
    )

    control = control.copy()
    treated = treated.copy()
    control['treatment'] = 0
    treated['treatment'] = 1
    trial = pd.concat([control, treated], ignore_index=True)
    covariates = [
        column for column in trial.columns
        if column not in ('time', 'censor', 'treatment')
    ]
    # Preserve the original tutorial calls: the Ye-style test receives the
    # configured assignment probability, while the prognostic variant uses its
    # default observed treatment fraction.
    adjusted = compute_adjusted_logrank_test(
        trial, covariates, return_details=True,
        p_trt=p_treated)
    if prognostic_score is None:
        prognostic_score = compute_martingale_score(trial, covariates)
    prognostic_logrank = prognostic_adjusted_logrank(
        trial, covariates, prognostic_score=prognostic_score)
    def fit_cox_pvalue(fit):
        try:
            return fit()
        except Exception as error:
            if not allow_cox_convergence_failure:
                raise
            try:
                from lifelines.exceptions import ConvergenceError
            except ImportError:
                raise error
            if not isinstance(error, ConvergenceError):
                raise
            warnings.warn(
                'A Cox test did not converge for an augmented-control analysis; '
                'saving its p-value as NaN.',
                RuntimeWarning,
                stacklevel=2,
            )
            return float('nan')

    cox_pvalue = fit_cox_pvalue(lambda: cox_adjusted_pvalue(trial, covariates))
    prognostic_cox_pvalue = fit_cox_pvalue(
        lambda: martingale_score_adjusted_pvalue(
            trial, covariates, prognostic_score=prognostic_score))

    def negative_log(p_value):
        return float(-np.log(np.clip(p_value, np.finfo(float).tiny, 1.0)))

    return {
        'log_pvalue_adjusted_logrank_init': adjusted['neg_log_p'],
        'log_pvalue_prog_adjusted_logrank_init': prognostic_logrank['log_pvalue'],
        'log_pvalue_cox_wald_x_init': negative_log(cox_pvalue),
        'log_pvalue_cox_wald_prog_init': negative_log(prognostic_cox_pvalue),
    }, prognostic_score


def analyze_one(reference_path, generators):
    import numpy as np
    import pandas as pd
    from metrics import fit_cox_model, general_metrics
    from simulations import compute_logrank_test
    with np.load(reference_path, allow_pickle=False) as archive:
        ref = {key: archive[key] for key in archive.files}
    meta = metadata(ref)
    if meta['schema_version'] != 1:
        raise ValueError(f'Unsupported schema: {reference_path}')
    config, seed, d = meta['config'], meta['mc_seed'], meta['fraction_index']
    count = config['n_generated_dataset']
    effects = config['treatment_effects']
    columns = ref['columns'].tolist()
    raw_columns = ref['raw_columns'].tolist()
    control = load_frame(ref, 'control', columns)
    control['treatment'] = 0
    raw_control = load_frame(ref, 'raw_control', raw_columns)
    feature_types = json.loads(str(ref['feature_types']))
    continuous = [ft['name'] for ft in feature_types if ft['type'] in ['pos', 'real']]
    categorical = [ft['name'] for ft in feature_types if ft['type'] == 'cat']
    generated, scores, control_logrank = {}, {}, {}
    with private_workspace():
        for generator in generators:
            path = reference_path.parent / (generator + '.npz')
            with np.load(path, allow_pickle=False) as archive:
                gen_meta = metadata(archive)
                if any(gen_meta.get(key) != value for key, value in meta.items()) or gen_meta['generator'] != generator:
                    raise ValueError(f'Mismatched reference/generator metadata: {path}')
                values = archive['data']
                if values.ndim != 3 or values.shape[0] != count or values.shape[2] != len(columns) - 1:
                    raise ValueError(f'Invalid synthetic dataset shape in {path}: {values.shape}')
                # Match the RNG state entering this generator's metrics in the original script.
                restore_rng(archive)
            frames = []
            for data in values:
                frame = pd.DataFrame(data, columns=columns[:-1])
                frame['treatment'] = 0
                frames.append(frame)
            generated[generator] = frames
            scores[generator] = general_metrics(
                control, frames, generator, include_nndr=True, include_tableone_min_p_value=True,
                categorical=categorical, continuous=continuous, nonnormal=continuous)[METRICS]
            if len(scores[generator]) != count:
                raise ValueError(f'Unexpected metric row count for {generator}')
            # This comparison is invariant across treatment effects (sixfold saving by default).
            control_logrank[generator] = [compute_logrank_test(frame, raw_control) for frame in frames]
    # Restore the full control-arm size with synthetic controls for experiment D.
    # Use a generator-specific, synthetic-draw-specific seed so results are stable
    # and independent of the analysis RNG state.
    raw_treated_for_size = load_frame(ref, 'raw_treated_0', raw_columns)
    synthetic_control_count = max(0, len(raw_treated_for_size) - len(raw_control))
    real_control_decoded = control.drop(columns=['treatment']).copy()
    augmented_controls = {}
    for generator_index, generator in enumerate(generators):
        augmented_controls[generator] = []
        for synthetic_id, frame in enumerate(generated[generator]):
            synthetic_control = frame.drop(columns=['treatment']).copy()
            if synthetic_control_count > len(synthetic_control):
                raise ValueError(
                    f'{generator} generated {len(synthetic_control)} controls, '
                    f'but D requires {synthetic_control_count}.'
                )
            rng = np.random.default_rng(
                np.random.SeedSequence([seed, d, generator_index, synthetic_id])
            )
            selected = rng.choice(len(synthetic_control), size=synthetic_control_count,
                                  replace=False)
            augmented_controls[generator].append(pd.concat(
                [real_control_decoded, synthetic_control.iloc[selected]],
                ignore_index=True,
            ))
    rows = []
    cached_prognostic_score = None
    cached_treated_covariates = None
    d_prognostic_scores = {generator: [None] * count for generator in generators}
    d_treated_covariates_cache = {generator: [None] * count for generator in generators}
    for t, effect in enumerate(effects):
        # Legacy true_univ_coef draws its seed from the ambient NumPy RNG.
        # Subsequent effects start from the state left by the treated simulation.
        if t:
            restore_rng(ref, f'treated_rng_{t - 1}')
        reference_seed = int(np.random.randint(1000))
        coef_true = reference_coefficient(effect, reference_seed, json.dumps(config, sort_keys=True))
        treated = load_frame(ref, f'treated_{t}', columns)
        treated['treatment'] = 1
        raw_treated = load_frame(ref, f'raw_treated_{t}', raw_columns)
        d_treated = treated.drop(columns=['treatment']).copy()
        d_treated_covariates = d_treated.drop(columns=['time', 'censor']).to_numpy()
        covariate_columns = [
            column for column in raw_columns if column not in ('time', 'censor')
        ]
        treated_covariates = raw_treated[covariate_columns].to_numpy()
        reusable_score = (
            cached_prognostic_score
            if cached_treated_covariates is not None
            and np.array_equal(treated_covariates, cached_treated_covariates)
            else None
        )
        procova, prognostic_score = procova_baselines(
            raw_control, raw_treated, config['p_treated'], reusable_score)
        if reusable_score is None:
            cached_prognostic_score = prognostic_score
            cached_treated_covariates = treated_covariates.copy()
        cox_columns = ['time', 'censor', 'treatment']
        coef, _, _, se = fit_cox_model(pd.concat([control, treated], ignore_index=True), cox_columns)
        result = pd.DataFrame({
            'XP_num': (seed - 1) * len(effects) * len(config['control_fractions']) + d * len(effects) + t,
            'D_control': raw_control['censor'].sum() * (len(raw_treated) / len(raw_control)),
            'D_treated': raw_treated['censor'].sum(), 'H0_coef_univ': coef_true,
            'H0_coef': effect, 'aug_perc': config['control_fractions'][d],
            'log_pvalue_init': compute_logrank_test(control, treated),
            'est_cox_coef_init': coef[0], 'est_cox_coef_se_init': se[0],
            # Explicit clustering keys: synthetic draws from one model are not independent MC runs.
            'MC_seed': seed, 'synthetic_id': np.arange(count),
            **procova,
        })
        additions = {}
        for generator in generators:
            logrank, coefficients, errors = [], [], []
            d_adjusted_logrank = []
            d_prognostic_logrank = []
            d_cox_wald = []
            d_prognostic_cox_wald = []
            for synthetic_id, frame in enumerate(generated[generator]):
                logrank.append(compute_logrank_test(frame, raw_treated))
                coef, _, _, se = fit_cox_model(pd.concat([frame, treated], ignore_index=True), cox_columns)
                coefficients.append(coef[0])
                errors.append(se[0])
                reusable_d_score = d_prognostic_scores[generator][synthetic_id]
                cached_d_covariates = d_treated_covariates_cache[generator][synthetic_id]
                if (cached_d_covariates is not None
                        and not np.array_equal(cached_d_covariates, d_treated_covariates)):
                    reusable_d_score = None
                procova_d, d_score = procova_baselines(
                    augmented_controls[generator][synthetic_id], d_treated,
                    config['p_treated'], reusable_d_score,
                    allow_cox_convergence_failure=True)
                if reusable_d_score is None:
                    d_prognostic_scores[generator][synthetic_id] = d_score
                    d_treated_covariates_cache[generator][synthetic_id] = (
                        d_treated_covariates.copy()
                    )
                d_adjusted_logrank.append(procova_d['log_pvalue_adjusted_logrank_init'])
                d_prognostic_logrank.append(procova_d['log_pvalue_prog_adjusted_logrank_init'])
                d_cox_wald.append(procova_d['log_pvalue_cox_wald_x_init'])
                d_prognostic_cox_wald.append(procova_d['log_pvalue_cox_wald_prog_init'])
            additions[f'log_pvalue_{generator}'] = logrank
            additions[f'log_pvalue_control_{generator}'] = control_logrank[generator]
            additions[f'est_cox_coef_{generator}'] = coefficients
            additions[f'est_cox_coef_se_{generator}'] = errors
            additions[f'log_pvalue_adjusted_logrank_D_{generator}'] = d_adjusted_logrank
            additions[f'log_pvalue_prog_adjusted_logrank_D_{generator}'] = d_prognostic_logrank
            additions[f'log_pvalue_cox_wald_x_D_{generator}'] = d_cox_wald
            additions[f'log_pvalue_cox_wald_prog_D_{generator}'] = d_prognostic_cox_wald
            for metric in METRICS:
                additions[f'{metric}_{generator}'] = scores[generator][metric].to_numpy()
        rows.append(pd.concat([result, pd.DataFrame(additions)], axis=1))
    return pd.concat(rows, ignore_index=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--input-dir', type=Path, default=DEFAULT_DATASET / 'generated_two_stage')
    result.add_argument(
        '--generators', nargs='+', choices=ANALYSIS_GENERATORS,
        help='Generators to analyze. Omit to discover every supported generator in the first MC archive.',
    )
    result.add_argument('--mc-start', type=int, default=1)
    result.add_argument('--n-mc', type=int, default=10)
    result.add_argument('--output', type=Path, required=True, help='Destination CSV (atomically replaced on success).')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if min(args.mc_start, args.n_mc) < 1:
        raise ValueError('MC start and count must be positive.')
    if args.generators is not None and len(set(args.generators)) != len(args.generators):
        raise ValueError('Generator names must be unique.')
    import numpy as np
    root = args.input_dir.resolve()
    first = root / f'MC_{args.mc_start:06d}' / 'control_0' / 'reference.npz'
    with np.load(first, allow_pickle=False) as archive:
        config = metadata(archive)['config']
    if args.generators is None:
        generators = [
            generator for generator in ANALYSIS_GENERATORS
            if (first.parent / f'{generator}.npz').is_file()
        ]
        if not generators:
            raise FileNotFoundError(
                f'No supported generator archives were found in {first.parent}. '
                'Pass --generators explicitly or check --input-dir.'
            )
        print(f'Discovered generators: {generators}', flush=True)
    else:
        generators = args.generators
    references = []
    # Validate the whole requested range before expensive metrics; never silently omit failures.
    for d in range(len(config['control_fractions'])):
        for seed in range(args.mc_start, args.mc_start + args.n_mc):
            path = root / f'MC_{seed:06d}' / f'control_{d}' / 'reference.npz'
            expected = {'schema_version': 1, 'config': config, 'mc_seed': seed, 'fraction_index': d}
            with np.load(path, allow_pickle=False) as archive:
                if metadata(archive) != expected:
                    raise ValueError(f'Incompatible simulation configuration: {path}')
            for generator in generators:
                generated_path = path.parent / (generator + '.npz')
                with np.load(generated_path, allow_pickle=False) as archive:
                    info = metadata(archive)
                    if any(info.get(key) != value for key, value in expected.items()) or info['generator'] != generator:
                        raise ValueError(f'Incompatible generator archive: {generated_path}')
            references.append(path)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Stream each MC/fraction block so memory does not grow with N_MC.
    with tempfile.NamedTemporaryFile(mode='w', dir=output.parent, suffix='.csv', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            offset = 0
            for path in references:
                print(f'Analyzing {path.parent}', flush=True)
                result = analyze_one(path, generators)
                result.index = range(offset, offset + len(result))
                result.to_csv(stream, header=offset == 0)
                offset += len(result)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
    print(f'Saved {offset} rows to {output}', flush=True)


if __name__ == '__main__':
    main()
