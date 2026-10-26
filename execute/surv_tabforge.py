"""Naive joint generation of covariates, observed time and event with TabFORGE.

Install the optional backend with ``pip install tabforge-ml`` (Python >= 3.10).
Time is an ordinary numerical feature and event an ordinary categorical
feature: this models observed records, without a censored survival likelihood.
The default pretrained backend downloads weights and performs dataset-specific
training. See https://github.com/SilenceX12138/TabFORGE for backend configuration.
"""
from numbers import Integral
from time import perf_counter

import numpy as np
import pandas as pd
import torch

_SURV_TYPES = ('surv', 'surv_weibull', 'surv_piecewise')
_MIN_TIME = 1e-3


def _categorical_columns(columns, event, time, feat_types_dict):
    names = {event}
    if feat_types_dict is not None:
        offset = 0
        for feature in feat_types_dict:
            width = 2 if feature['type'] in _SURV_TYPES else 1
            if offset + width > len(columns):
                raise ValueError('feat_types_dict does not match the expanded columns.')
            if feature['type'] in ('cat', 'ordinal'):
                names.add(columns[offset])
            offset += width
        if offset != len(columns):
            raise ValueError('feat_types_dict does not match the expanded columns.')
    if time in names:
        raise ValueError('Observed time must be numerical, not categorical.')
    return [name for name in columns if name in names]


def _record_time(profile, key, started):
    if profile is not None:
        profile[key] = profile.get(key, 0.0) + perf_counter() - started


def run(data, columns, target_column, time_to_event_column, n_generated_dataset,
        n_generated_sample=None, params=None, cond_gen=None, apply_rounding=False,
        feat_types_dict=None, mode='naive', profile=None):
    """Fit once and return a list of CPU float32 synthetic tensors.

    The interface follows ``surv_tabpfn.run``. Input is decoded numeric data
    (not one-hot), with columns in expanded feature-descriptor order. The event
    indicator must be 0/1 and time positive. Only ``mode='naive'`` is supported;
    conditional generation is not implemented and is explicitly rejected.

    ``params`` accepts ``random_state`` (default 1), ``checkpoint`` (default
    'pretrained'; None trains from scratch), ``detokeniser`` (default 'auto'),
    and TabFORGE constructor options: ``embedding_config``,
    ``architecture_config``, ``diffusion_config``, ``training_config``,
    ``runtime_config``, ``n_reference_neighbors``, ``observation_masking``,
    ``logger``. Configurations may be dictionaries or TabFORGE config objects.
    For example, ``runtime_config={'device': 'cuda'}`` selects a GPU.
    Each generated dataset receives a distinct deterministic seed.

    ``apply_rounding`` reuses the project's real-data precision rules and
    requires ``feat_types_dict``. Positive time and binary event guards are
    applied after rounding. ``profile`` optionally accumulates stage timings.
    """
    if mode != 'naive':
        raise ValueError("TabFORGE supports only mode='naive'.")
    if cond_gen is not None:
        raise NotImplementedError('Conditional TabFORGE generation is not implemented.')
    columns = list(columns)
    if len(set(columns)) != len(columns):
        raise ValueError('Column names must be unique.')
    if (target_column == time_to_event_column or target_column not in columns
            or time_to_event_column not in columns):
        raise ValueError('Specify distinct event and time columns present in columns.')
    values = data.detach().cpu().numpy() if torch.is_tensor(data) else np.asarray(data)
    if values.ndim != 2 or values.shape[1] != len(columns) or len(values) == 0:
        raise ValueError('data must be a nonempty matrix matching columns.')
    if not np.isfinite(values).all():
        raise ValueError('Training data must contain only finite numeric values.')
    df = pd.DataFrame(values, columns=columns)
    if not df[target_column].isin([0, 1]).all():
        raise ValueError('The event indicator must contain only 0 and 1.')
    if not (df[time_to_event_column] > 0).all():
        raise ValueError('Observed times must be strictly positive.')
    if n_generated_sample is None:
        n_generated_sample = len(df)
    for name, value in [('n_generated_dataset', n_generated_dataset),
                        ('n_generated_sample', n_generated_sample)]:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f'{name} must be a positive integer.')
    categorical = _categorical_columns(columns, target_column, time_to_event_column,
                                       feat_types_dict)
    if apply_rounding and feat_types_dict is None:
        raise ValueError('apply_rounding requires feat_types_dict.')
    options = dict(params or {})
    seed = options.pop('random_state', 1)
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError('random_state must be a nonnegative integer.')
    checkpoint = options.pop('checkpoint', 'pretrained')
    detokeniser = options.pop('detokeniser', 'auto')
    allowed = {'embedding_config', 'architecture_config', 'diffusion_config',
               'training_config', 'runtime_config', 'n_reference_neighbors',
               'observation_masking', 'logger'}
    if set(options) - allowed:
        raise ValueError(f'Unknown TabFORGE parameters: {sorted(set(options) - allowed)}')
    
    from tabforge import TabFORGEGenerator
    
    started = perf_counter()
    model = TabFORGEGenerator(task='unsupervision', categorical_features=categorical,
                             random_state=int(seed), **options)
    model.fit(df, checkpoint=checkpoint, detokeniser=detokeniser)
    _record_time(profile, 'tabforge_fit_seconds', started)

    started = perf_counter()
    if apply_rounding:
        # Works with both package imports and the existing execute/utils sys.path setup.
        try:
            from utils import data_processing
        except ModuleNotFoundError:
            import data_processing
        rounding_plan = data_processing.prepare_rounding_plan(values, feat_types_dict)
    _record_time(profile, 'rounding_setup_seconds', started)
    results = []
    for draw in range(n_generated_dataset):
        started = perf_counter()
        generated = model.generate(int(n_generated_sample), random_state=int(seed) + draw + 1)
        _record_time(profile, 'tabforge_sampling_seconds', started)
        started = perf_counter()
        if isinstance(generated, pd.DataFrame):
            if not generated.columns.is_unique or set(generated.columns) != set(columns):
                raise RuntimeError('TabFORGE returned an unexpected column schema.')
            generated = generated.loc[:, columns].to_numpy(dtype=np.float32)
        out = torch.tensor(np.asarray(generated, dtype=np.float32), dtype=torch.float32)
        if out.shape != (n_generated_sample, len(columns)) or not torch.isfinite(out).all():
            raise RuntimeError('TabFORGE returned an invalid shape or non-finite values.')
        if apply_rounding:
            out = data_processing.round_data_gen(values, out, feat_types_dict,
                                                 rounding_plan=rounding_plan)
        out[:, columns.index(target_column)] = out[:, columns.index(target_column)].round().clamp(0, 1)
        out[:, columns.index(time_to_event_column)] = out[:, columns.index(time_to_event_column)].clamp(min=_MIN_TIME)
        _record_time(profile, 'postprocessing_seconds', started)
        results.append(out)
    return results
