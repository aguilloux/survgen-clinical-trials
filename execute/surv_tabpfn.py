"""
TabPFN-based synthetic survival data generator.

TabPFN (Prior Labs) has no native notion of right-censoring, so this module
offers three strategies for handling the (time, event) pair, selected via
`mode`:

- mode="naive": (time, event) are fed to `TabPFNUnsupervisedModel` as plain
  continuous / categorical columns, with no censoring awareness at all.
  Censored rows are treated as if their observed time were a true event
  time.

- mode="uncensoring": mirrors synthcity's `SurvivalGANPlugin` "uncensoring"
  strategy (`synthcity/plugins/survival_analysis/_survival_pipeline.py`) and
  the min/indicator collapse used in
  `utils/data_processing.survival_variables_transformation`. A time-to-event
  model is fit on event-observed rows only and used to impute a plausible
  "true" event time for every censored row; `TabPFNUnsupervisedModel` is
  then trained on covariates + this fully-observed time column (no event
  column at all). At generation time, a separately fitted classifier
  re-injects censoring by predicting the event indicator for each synthetic
  row from its synthetic covariates; the generated time value itself is left
  unchanged (matches synthcity's own default behaviour, keeping the two
  comparable). In practice `time` folded into TabPFN's undifferentiated
  column-by-column joint sampling gets only a weak covariate->time signal on
  small real datasets -- see mode="survival_function" below.

- mode="survival_function": mirrors `SurvivalPipeline`'s actual *default*
  strategy (`strategy="survival_function"`, verified directly against the
  installed synthcity source). TabPFN generates covariates *only* (time and
  event are dropped from the joint fit entirely); the event indicator is
  then sampled (from the real censoring rate, or a fitted classifier); and
  `time` is computed by a dedicated, censoring-aware time-to-event model
  (`synthcity.plugins.core.models.time_to_event`, e.g. "weibull_aft" or the
  DeepHit-based "survival_function_regression") via its `predict_any(X,
  event)`, conditioned on the synthetic covariates and sampled event status.
  This decouples "what do covariates look like" (TabPFN's job) from "what
  does time-to-event look like given covariates and censoring" (the TTE
  model's job, trained with a proper censored likelihood) -- recommended
  over "uncensoring" for real datasets.

All three modes share the same `run(...)` signature as `execute/surv_gan.py`
/ `execute/surv_vae.py` so this module drops into the same
generator-comparison loop unchanged.
"""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions import unsupervised

module_path = Path.cwd().parent / 'utils'
sys.path.append(str(module_path))
import data_processing

_SURV_TYPES = ('surv', 'surv_weibull', 'surv_loglog', 'surv_piecewise')
_MIN_TIME = 1e-3  # survival loaders (e.g. synthcity) reject time <= 0


def set_seed(seed=1):
    random.seed(seed)                            # Python built-in
    np.random.seed(seed)                         # NumPy
    torch.manual_seed(seed)                      # PyTorch (CPU)


def _categorical_indices(columns, target_column, feat_types_dict):
    """
    Column indices (into `columns`) that TabPFN should treat as categorical:
    every 'cat'/'ordinal' feature from `feat_types_dict`, plus the event
    indicator column. `feat_types_dict` walks the *original* (pre-expansion)
    variable list, so survival types are stepped over two-at-a-time exactly
    like `data_processing.round_data_gen` does, to stay aligned with
    `columns` (which is the expanded [..., 'time', 'censor', ...] list).
    """
    cat_names = {target_column}
    if feat_types_dict is not None:
        feat_idx = 0
        for d in feat_types_dict:
            width = 2 if d['type'] in _SURV_TYPES else 1
            if d['type'] in ('cat', 'ordinal'):
                cat_names.add(columns[feat_idx])
            feat_idx += width
    return [i for i, name in enumerate(columns) if name in cat_names]


def _build_unsupervised_model(params=None):
    params = params or {}
    clf = TabPFNClassifier(device=params.get('device', 'auto'), n_estimators=params.get('n_estimators', 'auto'))
    reg = TabPFNRegressor(device=params.get('device', 'auto'), n_estimators=params.get('n_estimators', 'auto'))
    return unsupervised.TabPFNUnsupervisedModel(tabpfn_clf=clf, tabpfn_reg=reg)


def _fit_uncensoring_model(Xcov, T, E, method='tabpfn', params=None):
    """
    Fit a time-to-event model to impute a plausible event time for censored
    rows. Returns a callable `predict(X) -> np.ndarray` of imputed times.
    """
    params = params or {}
    if method == 'tabpfn':
        model = TabPFNRegressor(device=params.get('device', 'auto'))
        model.fit(Xcov[E == 1], T[E == 1])
        return lambda X: np.asarray(model.predict(X))
    elif method == 'weibull_aft':
        # Fallback backend: already-installed synthcity dependency, thin
        # wrapper around lifelines.WeibullAFTFitter. Fit on all rows (it is
        # itself censoring-aware), then only its predictions for censored
        # rows are used.
        from synthcity.plugins.core.models.time_to_event import get_model_template
        tte_model = get_model_template('weibull_aft')()
        tte_model.fit(Xcov, T, E)
        return lambda X: np.asarray(tte_model.predict(X))
    raise ValueError(f"Unknown uncensoring_model {method!r}, expected 'tabpfn' or 'weibull_aft'")


def _fit_tte_model(Xcov, T, E, method='survival_function_regression'):
    """
    Fit a censoring-aware time-to-event model exposing `predict_any(X, E) ->
    time`, i.e. "what would time-to-event look like for this covariate row,
    were it censored (E=0) or not (E=1)". Used by mode="survival_function" to
    compute time for every synthetic row from a model trained with a proper
    censored likelihood, rather than from TabPFN's undifferentiated joint
    column sampling.
    """
    from synthcity.plugins.core.models.time_to_event import get_model_template
    tte_model = get_model_template(method)()
    tte_model.fit(Xcov, T, E)
    return tte_model


def _generate_once(model, n_samples, params):
    params = params or {}
    return model.generate_synthetic_data(
        n_samples=n_samples,
        t=params.get('t', 1.0),
        n_permutations=params.get('n_permutations', 3),
    )


def run(data, columns, target_column, time_to_event_column, n_generated_dataset,
        n_generated_sample=None, params=None, cond_gen=None, apply_rounding=False,
        feat_types_dict=None, mode='naive'):
    """
    Use TabPFN for survival data generation.

    Parameters
    ----------
    data : torch.Tensor
        Decoded (non-one-hot) dataset, one column per `columns` entry -- the
        same representation `surv_gan.run`/`surv_vae.run` expect.
    columns : list of str
        Column names for `data`, in the same order as `feat_types_dict`
        after survival-type expansion (see `data_processing.read_data`).
    target_column : str
        Name of the event/censoring indicator column (1 = event, 0 = censored).
    time_to_event_column : str
        Name of the observed time column.
    n_generated_dataset : int
        Number of independent synthetic datasets to generate.
    n_generated_sample : int, optional
        Number of rows per generated dataset. Defaults to `data.shape[0]`.
    params : dict, optional
        Generation knobs: `t` (temperature, default 1.0), `n_permutations`
        (default 3), `n_estimators` / `device` (TabPFN model knobs); for
        mode="uncensoring": `uncensoring_model` ("tabpfn" default or
        "weibull_aft"); for mode="survival_function": `uncensoring_model`
        (time-to-event backend, "survival_function_regression" default or
        "weibull_aft") and `censoring_strategy` ("random" default, matching
        the real censoring rate, or "covariate_dependent", from a fitted
        classifier).
    cond_gen : not supported
        TabPFNUnsupervisedModel has no native conditional-generation API.
        Passing a non-None value raises NotImplementedError.
    apply_rounding : bool, default=False
        If True, round generated columns onto the real-data precision grid
        via `data_processing.round_data_gen` (unchanged, reused as-is).
    feat_types_dict : list of dict, optional
        Original (pre-expansion) type descriptors, as returned by
        `data_processing.read_data`. Needed to infer categorical columns and
        for `apply_rounding`.
    mode : {"naive", "uncensoring", "survival_function"}, default="naive"
        See module docstring.

    Returns
    -------
    list of torch.Tensor
        `n_generated_dataset` synthetic datasets, each of shape
        (n_generated_sample, len(columns)), columns in the same order as
        `columns`.
    """
    if cond_gen is not None:
        raise NotImplementedError(
            "TabPFNUnsupervisedModel has no native conditional-generation API; "
            "cond_gen is not supported."
        )
    if mode not in ('naive', 'uncensoring', 'survival_function'):
        raise ValueError(f"Unknown mode {mode!r}, expected 'naive', 'uncensoring' or 'survival_function'")

    set_seed()
    params = params or {}

    df = pd.DataFrame(data.numpy(), columns=columns)  # Preprocessed dataset
    if n_generated_sample is None:
        n_generated_sample = df.shape[0]

    cat_indices = _categorical_indices(columns, target_column, feat_types_dict)
    target_idx = columns.index(target_column)
    time_idx = columns.index(time_to_event_column)

    est_data_gen = []

    if mode == 'naive':
        model = _build_unsupervised_model(params)
        model.set_categorical_features(cat_indices)
        model.fit(df)

        for _ in range(n_generated_dataset):
            out = _generate_once(model, n_generated_sample, params)
            out[:, target_idx] = out[:, target_idx].round().clamp(0, 1)
            # Strictly positive: survival loaders (e.g. synthcity's
            # SurvivalAnalysisDataLoader) reject time <= 0.
            out[:, time_idx] = out[:, time_idx].clamp(min=_MIN_TIME)
            if apply_rounding:
                out = data_processing.round_data_gen(df.values, out, feat_types_dict)
            est_data_gen.append(out)

    elif mode == 'uncensoring':
        Xcov = df.drop(columns=[target_column, time_to_event_column])
        T = df[time_to_event_column].copy()
        E = df[target_column]

        uncensor_predict = _fit_uncensoring_model(
            Xcov, T, E, method=params.get('uncensoring_model', 'tabpfn'), params=params,
        )
        censored = (E == 0).values
        if censored.any():
            T.iloc[censored] = uncensor_predict(Xcov[censored])

        df_train = Xcov.copy()
        df_train[time_to_event_column] = T
        train_columns = list(df_train.columns)
        train_cat_indices = [
            train_columns.index(columns[i]) for i in cat_indices if columns[i] in train_columns
        ]

        model = _build_unsupervised_model(params)
        model.set_categorical_features(train_cat_indices)
        model.fit(df_train)

        censoring_clf = TabPFNClassifier(device=params.get('device', 'auto'))
        censoring_clf.fit(Xcov, E)

        train_time_idx = train_columns.index(time_to_event_column)
        for _ in range(n_generated_dataset):
            gen = _generate_once(model, n_generated_sample, params)
            gen[:, train_time_idx] = gen[:, train_time_idx].clamp(min=_MIN_TIME)
            gen_df = pd.DataFrame(gen.numpy(), columns=train_columns)
            event_syn = censoring_clf.predict(gen_df[list(Xcov.columns)])

            out_df = gen_df.copy()
            out_df[target_column] = event_syn
            out_df = out_df[columns]  # restore the original column order
            out = torch.from_numpy(out_df.to_numpy(dtype='float32'))
            if apply_rounding:
                out = data_processing.round_data_gen(df.values, out, feat_types_dict)
            est_data_gen.append(out)

    else:  # mode == 'survival_function'
        Xcov = df.drop(columns=[target_column, time_to_event_column])
        T = df[time_to_event_column]
        E = df[target_column]
        censoring_ratio = float((E == 0).mean())
        censoring_strategy = params.get('censoring_strategy', 'random')
        if censoring_strategy not in ('random', 'covariate_dependent'):
            raise ValueError(
                f"Unknown censoring_strategy {censoring_strategy!r}, "
                "expected 'random' or 'covariate_dependent'"
            )

        tte_model = _fit_tte_model(Xcov, T, E, method=params.get('uncensoring_model', 'survival_function_regression'))

        censoring_clf = None
        if censoring_strategy == 'covariate_dependent':
            censoring_clf = TabPFNClassifier(device=params.get('device', 'auto'))
            censoring_clf.fit(Xcov, E)

        # Covariates-only joint generator -- time/event never enter its
        # training data, so the generic column-by-column sampler cannot
        # dilute the covariate distribution with a weakly-learned
        # covariate->time relationship (that job belongs to tte_model).
        cov_columns = list(Xcov.columns)
        cov_cat_indices = [
            cov_columns.index(columns[i]) for i in cat_indices if columns[i] in cov_columns
        ]
        model = _build_unsupervised_model(params)
        model.set_categorical_features(cov_cat_indices)
        model.fit(Xcov)

        for _ in range(n_generated_dataset):
            cov_gen = _generate_once(model, n_generated_sample, params)
            cov_gen_df = pd.DataFrame(cov_gen.numpy(), columns=cov_columns)

            if censoring_strategy == 'covariate_dependent':
                event_syn = np.asarray(censoring_clf.predict(cov_gen_df), dtype='float32')
            else:
                event_syn = (np.random.rand(len(cov_gen_df)) >= censoring_ratio).astype('float32')
            event_syn_series = pd.Series(event_syn, index=cov_gen_df.index)

            time_syn = np.asarray(tte_model.predict_any(cov_gen_df, event_syn_series), dtype='float32')
            time_syn = np.clip(time_syn, a_min=_MIN_TIME, a_max=None)

            out_df = cov_gen_df.copy()
            out_df[time_to_event_column] = time_syn
            out_df[target_column] = event_syn
            out_df = out_df[columns]  # restore the original column order
            out = torch.from_numpy(out_df.to_numpy(dtype='float32'))
            if apply_rounding:
                out = data_processing.round_data_gen(df.values, out, feat_types_dict)
            est_data_gen.append(out)

    return est_data_gen
