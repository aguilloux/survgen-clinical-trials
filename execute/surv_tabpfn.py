"""
TabPFN-based synthetic survival data generator.

TabPFN (Prior Labs) has no native notion of right-censoring, so this module
offers two strategies for handling the (time, event) pair, selected via `mode`:

- mode="naive": (covariates, time, event) are all fed to `TabPFNUnsupervisedModel` as plain continuous / categorical columns, 
  with no censoring awareness at all. Censored rows are treated as if their observed time were a true event time. 
  The only post-hoc guards are rounding the event indicator to {0, 1} and clamping time strictly positive.

- mode="survival_function": mirrors synthcity `SurvivalPipeline`'s actual *default* strategy (`strategy="survival_function"`, 
  verified directly against the installed synthcity source). It decouples "what do covariates look like" (TabPFN's job) 
  from "what does time-to-event look like given covariates and censoring" (a dedicated time-to-event (TTE) model trained with a proper censored likelihood):

    1. TabPFN generates covariates only -- `time` never enters the joint fit, so the generic column-by-column sampler cannot dilute the covariate
       distribution with a weakly-learned covariate->time relationship. The event indicator enters the joint fit only for `censoring_strategy="joint"`.
    2. The synthetic event indicator is set per `params["censoring_strategy"]`:
         - "random" (default): drawn from the real censoring rate, independent of covariates.
         - "covariate_dependent": predicted from the synthetic covariates by a `TabPFNClassifier` fitted on the real (covariates -> event).
         - "joint": the event column is added to TabPFN's joint fit and the value TabPFN sampled is kept as-is, so covariate<->censoring
           dependence is carried by the generator itself (the closest analogue of survGAN keeping the event label it was conditioned on).
    3. `time` is computed for every synthetic row by the TTE model via `predict_any(X, event)`, conditioned on the synthetic covariates and the
       sampled event status. Backends (`params["tte_model"]`):
         - "survival_function_regression" (default) / "weibull_aft": synthcity `time_to_event` templates.
         - "survivalpfn": rgklab's SurvivalPFN, an in-context PFN in the same no-training spirit as TabPFN (see `_SurvivalPFNTTE`).

Both modes share the same `run(...)` signature as `execute/surv_gan.py` /
`execute/surv_vae.py` so this module drops into the same generator-comparison loop unchanged.
"""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import inspect

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions import unsupervised

module_path = Path.cwd().parent / 'utils'
sys.path.append(str(module_path))
import data_processing

_SURV_TYPES = ('surv', 'surv_weibull', 'surv_piecewise')
_MIN_TIME = 1e-3  # survival loaders (e.g. synthcity) reject time <= 0
_MODES = ('naive', 'survival_function')
_CENSORING_STRATEGIES = ('random', 'covariate_dependent', 'joint', 'competing_times')


def set_seed(seed=1):
    random.seed(seed)                            # Python built-in
    np.random.seed(seed)                         # NumPy
    torch.manual_seed(seed)                      # PyTorch (CPU)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _categorical_indices(columns, target_column, feat_types_dict):
    """
    Column indices (into `columns`) that TabPFN should treat as categorical:
    every 'cat'/'ordinal' feature from `feat_types_dict`, plus the event indicator column. 
    `feat_types_dict` walks the *original* (pre-expansion) variable list, 
    so survival types are stepped over two-at-a-time exactly like `data_processing.round_data_gen` does, 
    to stay aligned with `columns` (which is the expanded [..., 'time', 'censor', ...] list).
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


def _build_unsupervised_model(params):
    clf = TabPFNClassifier(device=params.get('device', 'auto'), n_estimators=params.get('n_estimators', 'auto'))
    reg = TabPFNRegressor(device=params.get('device', 'auto'), n_estimators=params.get('n_estimators', 'auto'))
    return unsupervised.TabPFNUnsupervisedModel(tabpfn_clf=clf, tabpfn_reg=reg)


def _generate_once(model, n_samples, params):
    return model.generate_synthetic_data(
        n_samples=n_samples,
        t=params.get('t', 1.0),
        n_permutations=params.get('n_permutations', 3),
    )


# ---------------------------------------------------------------------------
# Time-to-event model for mode="survival_function"
# ---------------------------------------------------------------------------
def _survivalpfn_device(params):
    dev = params.get('survivalpfn_device')
    if dev:
        return dev
    return 'cuda' if torch.cuda.is_available() else 'cpu'

class _SurvivalPFNTTE:
    """
    Generative adapter around rgklab's SurvivalPFN.

    SurvivalPFN predicts a posterior predictive *event-time distribution*. For synthetic generation we sample from 
    that distribution instead of collapsing it to a median/mode/RMST point prediction.

    Two in-context estimators are fitted on the same observed `(X, T, E)` data:
      * `_event`: `delta=E`, modelling latent event time Te under right censoring.
      * `_cens`: `delta=1-E`, symmetrically treating censoring as the event and
        observed failures as right-censoring, modelling latent censoring time Tc.

    This supports two generation styles:
      * `predict_any(X, E)`: compatibility path for the existing censoring strategies. 
        It samples from the event-time posterior where E=1 and from the censoring-time posterior where E=0.
      * `sample_observed(X)`: competing-times path. It independently draws Te ~ p(Te|X, real context) and Tc ~ p(Tc|X, real context), 
        then returns T=min(Te,Tc) and E=1[Te <= Tc]. This keeps the synthetic observed time and event indicator mutually consistent by construction.

    Relevant `params` keys (all optional):
        survivalpfn_device          : "cpu" / "cuda" / "cuda:0"
        survivalpfn_model_path      : checkpoint path or HF repo id (default "shi-ang/SurvivalPFN")
        survivalpfn_model_censoring : bool, default True
        survivalpfn_calibrate       : bool, default False
        survivalpfn_sample_times    : bool, default True. If False, retain the old point-estimate behaviour.
        survivalpfn_point_estimate  : "median" (default), "mode" or "rmst"; used only when sample_times=False.
    """

    def __init__(self, device='cpu', model_path='shi-ang/SurvivalPFN',
                 point_estimate='median', model_censoring=True, calibrate=False,
                 sample_times=True, tail_cap=None, tail_cap_factor=10.0):
        self.device = device
        self.model_path = model_path
        self.point_estimate = point_estimate
        self.model_censoring = model_censoring
        self.calibrate = calibrate
        self.sample_times = sample_times
        # Upper bound on sampled times. Explicit `tail_cap` wins; otherwise it is
        # set in fit() to `tail_cap_factor * max observed time`, so exponential
        # right-tail draws (see _sample_histogram_distribution) cannot produce
        # times orders of magnitude past anything in the real data.
        self.tail_cap = tail_cap
        self.tail_cap_factor = tail_cap_factor
        self._event = None
        self._cens = None
        self._observed_has_censoring = False

    def _new_estimator(self):
        try:
            from survivalpfn import SurvivalEstimator
        except ImportError as exc:
            raise ImportError(
                "SurvivalPFN is not installed. Install it from https://github.com/rgklab/SurvivalPFN "
                "(`pip install git+https://github.com/rgklab/SurvivalPFN`)."
            ) from exc
        kwargs = {
            'device': self.device,
            'model_path': self.model_path,
        }
        # `device` and `model_path` are part of the public README API. Keep the
        # older optional `calibrate` knob only when the installed estimator
        # actually exposes it, so this adapter remains compatible with newer
        # SurvivalPFN versions.
        try:
            if 'calibrate' in inspect.signature(SurvivalEstimator).parameters:
                kwargs['calibrate'] = self.calibrate
        except (TypeError, ValueError):
            # Some callables do not expose an inspectable signature; in that
            # case stick to the public constructor arguments.
            pass
        return SurvivalEstimator(**kwargs)

    @staticmethod
    def _as_array(a):
        return np.asarray(getattr(a, 'values', a), dtype='float32')

    @staticmethod
    def _to_numpy_1d(x, expected_len, label):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        x = np.asarray(x, dtype='float32').reshape(-1)
        if x.shape[0] != expected_len:
            raise RuntimeError(f"SurvivalPFN {label} returned {x.shape[0]} values for {expected_len} query rows.")
        if not np.all(np.isfinite(x)):
            raise RuntimeError(f"SurvivalPFN {label} returned non-finite times.")
        return x

    def fit(self, Xcov, T, E):
        X, T, E = self._as_array(Xcov), self._as_array(T), self._as_array(E).ravel()
        self._observed_has_censoring = bool((E == 0).any())
        if self.tail_cap is None:
            self.tail_cap = float(np.max(T)) * float(self.tail_cap_factor)

        # Event-time model: standard right-censored survival problem.
        self._event = self._new_estimator().fit(X=X, delta=E, T=T)

        # Censoring-time model: censoring becomes the event of interest and
        # observed failures are right-censored observations of Tc.
        self._cens = None
        if self.model_censoring and self._observed_has_censoring:
            delta_cens = (E == 0).astype('float32')
            self._cens = self._new_estimator().fit(X=X, delta=delta_cens, T=T)
        return self

    def _sample_histogram_distribution(self, dist, expected_len, label):
        """Draw one time per query row from SurvivalPFN's HistogramDistribution.

        SurvivalPFN's histogram representation stores one categorical mass per
        finite time bin plus a final residual right-tail mass.  For a selected
        finite bin we sample uniformly inside that bin.  The residual tail has
        no finite upper edge, so we continue the hazard implied by the last
        finite bin as an exponential tail.  This is an explicit extrapolation
        assumption, not additional information learned by SurvivalPFN.
        """
        if not hasattr(dist, 'probs') or not hasattr(dist, 'bin_edges'):
            raise RuntimeError(
                "SurvivalPFN's predictive distribution exposes neither a native "
                "sampler nor the expected histogram attributes `probs` and "
                "`bin_edges`."
            )

        probs = torch.as_tensor(dist.probs).detach().to(dtype=torch.float32)
        edges = torch.as_tensor(dist.bin_edges).detach().to(dtype=torch.float32, device=probs.device)

        if probs.ndim == 1:
            probs = probs.unsqueeze(0)
        if probs.ndim != 2:
            raise RuntimeError(
                f"SurvivalPFN {label} histogram probabilities have unexpected "
                f"shape {tuple(probs.shape)}; expected [N, n_bins]."
            )
        n, n_bins = probs.shape
        if n != expected_len:
            raise RuntimeError(
                f"SurvivalPFN {label} histogram has {n} rows for {expected_len} query rows."
            )

        # `bin_edges` comes back as [N, n_bins] (the shared edge vector broadcast
        # over query rows). K categorical masses = K-1 finite bins + one residual
        # right-tail mass, with K edges: [e0,e1), ..., [e_{K-2},e_{K-1}), [e_{K-1}, inf).
        if edges.ndim == 1:
            edges = edges.unsqueeze(0).expand(n, -1)
        if tuple(edges.shape) != (n, n_bins):
            raise RuntimeError(
                f"SurvivalPFN {label} histogram: {n_bins} probability masses but "
                f"bin_edges shape {tuple(edges.shape)}; expected {(n, n_bins)}."
            )
        if n_bins < 2 or not torch.all(edges[:, 1:] > edges[:, :-1]):
            raise RuntimeError("SurvivalPFN histogram bin edges are not strictly increasing.")

        # Normalize defensively in case of tiny floating-point drift.
        probs = probs.clamp_min(0)
        row_sum = probs.sum(dim=1, keepdim=True)
        if torch.any(row_sum <= 0):
            raise RuntimeError("SurvivalPFN histogram contains a row with zero probability mass.")
        probs = probs / row_sum

        chosen = torch.distributions.Categorical(probs=probs).sample()  # [N]
        rows = torch.arange(n, device=probs.device)
        draws = torch.empty(n, device=probs.device, dtype=torch.float32)

        finite = chosen < (n_bins - 1)
        if finite.any():
            r, k = rows[finite], chosen[finite]
            lo = edges[r, k]
            hi = edges[r, k + 1]
            u = torch.rand(r.shape[0], device=probs.device)
            draws[finite] = lo + u * (hi - lo)

        tail = ~finite
        if tail.any():
            # Extrapolate beyond the final edge by continuing the constant
            # hazard implied by the last finite interval. For each row,
            # S(end)/S(start) = p_tail / (p_last_bin + p_tail).
            r = rows[tail]
            last_edge = edges[r, n_bins - 1]
            width = (edges[r, n_bins - 1] - edges[r, n_bins - 2]).clamp_min(1e-12)
            p_last = probs[r, n_bins - 2]
            p_tail = probs[r, n_bins - 1]
            ratio = (p_tail / (p_last + p_tail).clamp_min(1e-12)).clamp(1e-6, 1 - 1e-6)
            hazard = -torch.log(ratio) / width
            scale = torch.where(hazard > 1e-8, 1.0 / hazard, width)
            tail_extra = torch.distributions.Exponential(rate=1.0 / scale).sample()
            draws[tail] = last_edge + tail_extra

        if self.tail_cap is not None:
            draws = draws.clamp(max=float(self.tail_cap))

        return self._to_numpy_1d(draws, expected_len, label)

    def _draw_time(self, estimator, X, label):
        if estimator is None:
            raise RuntimeError(f"SurvivalPFN {label} model is not fitted.")

        if self.sample_times:
            dist = estimator.predict_event_distribution(X)

            # Prefer a native sampler if a future SurvivalPFN release adds one.
            sampler = getattr(dist, 'sample', None)
            if sampler is None:
                sampler = getattr(dist, 'rsample', None)
            if sampler is not None:
                draw = sampler()
                return self._to_numpy_1d(draw, len(X), label)

            # Current SurvivalPFN HistogramDistribution has no sample()/rsample(),
            # so sample explicitly from its public histogram representation.
            return self._sample_histogram_distribution(dist, len(X), label)

        draw = estimator.predict_event_time(X, type=self.point_estimate)
        return self._to_numpy_1d(draw, len(X), label)

    def predict_any(self, X, E):
        """Generate a time conditional on an externally generated E.

        This method keeps compatibility with `random`, `covariate_dependent`,
        and `joint`. With posterior sampling enabled, rows with E=1 receive an
        event-time draw and rows with E=0 receive a censoring-time draw.
        """
        X = self._as_array(X)
        E = self._as_array(E).ravel()
        if len(E) != len(X):
            raise ValueError("X and E must contain the same number of rows.")

        t_event = self._draw_time(self._event, X, 'event-time')
        if self._cens is None:
            # Backwards-compatible behaviour when censoring modelling is disabled
            # or the real dataset contains no censored observations.
            return t_event

        t_cens = self._draw_time(self._cens, X, 'censoring-time')
        return np.where(E == 1, t_event, t_cens).astype('float32')

    def sample_observed(self, X):
        """Sample a coherent observed `(T, E)` via competing latent times.

        Te and Tc are sampled independently conditional on X from the two
        SurvivalPFN posterior predictive distributions, then

            T = min(Te, Tc)
            E = 1[Te <= Tc].

        If the real context has no censored observations, there is no empirical
        censoring process to fit, so all generated rows are treated as events.
        """
        if not self.sample_times:
            raise ValueError(
                "sample_observed()/censoring_strategy='competing_times' requires "
                "survivalpfn_sample_times=True; point estimates would turn the "
                "censoring mechanism into a hard per-covariate threshold."
            )

        X = self._as_array(X)
        t_event = self._draw_time(self._event, X, 'event-time')

        if not self._observed_has_censoring:
            return t_event.astype('float32'), np.ones(len(X), dtype='float32')

        if not self.model_censoring or self._cens is None:
            raise ValueError(
                "censoring_strategy='competing_times' requires "
                "survivalpfn_model_censoring=True when the real data contain "
                "censored observations."
            )

        t_cens = self._draw_time(self._cens, X, 'censoring-time')
        event = (t_event <= t_cens).astype('float32')
        observed_time = np.minimum(t_event, t_cens).astype('float32')
        return observed_time, event

    


# class _SurvivalPFNTTE:
#     """
#     Adapter exposing the slice of synthcity's `TimeToEventPlugin` interface this
#     module needs (`fit(Xcov, T, E)`, `predict_any(X, E)`) on top of rgklab's
#     SurvivalPFN (https://github.com/rgklab/SurvivalPFN) -- an in-context PFN
#     that, like TabPFN, produces a posterior event-time distribution in a single
#     forward pass with no dataset-specific training.

#     An *event-time* estimator is fit on the real `(X, delta=E, T)`. When
#     `model_censoring=True` a symmetric *censoring-time* estimator is fit with
#     the indicator flipped (`delta = 1 - E`), so `predict_any` gives an E=0
#     synthetic row a plausible censoring time rather than an event time --
#     mirroring synthcity's "survival_function_regression" paired event/censoring
#     heads. With `model_censoring=False` the event-time estimate is used for
#     every row.

#     Relevant `params` keys (all optional):
#         survivalpfn_device          : "cpu" / "cuda" / "cuda:0" (default: cuda if available else cpu)
#         survivalpfn_model_path      : checkpoint path or HF repo id (default "shi-ang/SurvivalPFN")
#         survivalpfn_point_estimate  : "median" (default), "mode" or "rmst"
#         survivalpfn_model_censoring : bool, default True
#         survivalpfn_calibrate       : bool, default False (temperature calibration, slower)
#     """

#     def __init__(self, device='cpu', model_path='shi-ang/SurvivalPFN',
#                  point_estimate='median', model_censoring=True, calibrate=False):
#         self.device = device
#         self.model_path = model_path
#         self.point_estimate = point_estimate
#         self.model_censoring = model_censoring
#         self.calibrate = calibrate
#         self._event = None
#         self._cens = None

#     def _new_estimator(self):
#         try:
#             from survivalpfn import SurvivalEstimator
#         except ImportError as exc:
#             raise ImportError(
#                 "SurvivalPFN is not installed. Install it from "
#                 "https://github.com/rgklab/SurvivalPFN "
#                 "(`pip install git+https://github.com/rgklab/SurvivalPFN`)."
#             ) from exc
#         return SurvivalEstimator(device=self.device, model_path=self.model_path, calibrate=self.calibrate)

#     @staticmethod
#     def _as_array(a):
#         return np.asarray(getattr(a, 'values', a), dtype='float32')

#     def fit(self, Xcov, T, E):
#         X, T, E = self._as_array(Xcov), self._as_array(T), self._as_array(E)
#         self._event = self._new_estimator().fit(X=X, delta=E, T=T)
#         self._cens = None
#         if self.model_censoring and (E == 0).any():
#             self._cens = self._new_estimator().fit(X=X, delta=(E == 0).astype('float32'), T=T)
#         return self

#     def predict_any(self, X, E):
#         X = self._as_array(X)
#         E = self._as_array(E).ravel()
#         t_event = np.asarray(self._event.predict_event_time(X, type=self.point_estimate), dtype='float32')
#         if self._cens is None:
#             return t_event
#         t_cens = np.asarray(self._cens.predict_event_time(X, type=self.point_estimate), dtype='float32')
#         return np.where(E == 1, t_event, t_cens)


def _fit_tte_model(Xcov, T, E, method='survival_function_regression', params=None):
    """
    Fit a censoring-aware time-to-event model exposing `predict_any(X, E) -> time`,
    i.e. "what would time-to-event look like for this covariate row, were it
    censored (E=0) or not (E=1)". Used by mode="survival_function" to compute
    time for every synthetic row from a model trained with a proper censored
    likelihood, rather than from TabPFN's undifferentiated joint column sampling.

    `method="survivalpfn"` routes to `_SurvivalPFNTTE`; any other value is a
    synthcity `time_to_event` template name ("survival_function_regression"
    default, or "weibull_aft").
    """
    params = params or {}
    if method == 'survivalpfn':
        tte_model = _SurvivalPFNTTE(
            device=_survivalpfn_device(params),
            model_path=params.get('survivalpfn_model_path', 'shi-ang/SurvivalPFN'),
            point_estimate=params.get('survivalpfn_point_estimate', 'median'),
            model_censoring=params.get('survivalpfn_model_censoring', True),
            calibrate=params.get('survivalpfn_calibrate', False),
            sample_times=params.get('survivalpfn_sample_times', True),
            tail_cap=params.get('survivalpfn_tail_cap'),
        )
        tte_model.fit(Xcov, T, E)
        return tte_model

    from synthcity.plugins.core.models.time_to_event import get_model_template
    tte_model = get_model_template(method)()
    tte_model.fit(Xcov, T, E)
    return tte_model


# ---------------------------------------------------------------------------
# Mode implementations
# ---------------------------------------------------------------------------
def _run_naive(df, columns, target_column, time_to_event_column, cat_indices,
               n_generated_dataset, n_generated_sample, params, apply_rounding, feat_types_dict):
    """
    mode="naive": TabPFN jointly samples covariates + time + event as ordinary columns, with no censoring awareness. 
    Post-hoc guards only: event rounded to {0, 1}, time clamped strictly positive.
    """
    target_idx = columns.index(target_column)
    time_idx = columns.index(time_to_event_column)

    model = _build_unsupervised_model(params)
    model.set_categorical_features(cat_indices)
    model.fit(df)

    est_data_gen = []
    for _ in range(n_generated_dataset):
        out = _generate_once(model, n_generated_sample, params)
        out[:, target_idx] = out[:, target_idx].round().clamp(0, 1)
        out[:, time_idx] = out[:, time_idx].clamp(min=_MIN_TIME)
        if apply_rounding:
            out = data_processing.round_data_gen(df.values, out, feat_types_dict)
        est_data_gen.append(out)
    return est_data_gen


def _sample_synthetic_events(censoring_strategy, gen_df, cov_gen_df, target_column, censoring_clf, censoring_ratio):
    """Synthetic event indicator for mode="survival_function", per `censoring_strategy`."""
    if censoring_strategy == 'joint':
        return gen_df[target_column].round().clip(0, 1).to_numpy(dtype='float32')
    if censoring_strategy == 'covariate_dependent':
        # return np.asarray(censoring_clf.predict(cov_gen_df), dtype='float32')
        event_proba = np.asarray(censoring_clf.predict_proba(cov_gen_df))
        classes = np.asarray(censoring_clf.classes_)
        event_class_idx = np.flatnonzero(classes == 1)
        if len(event_class_idx) != 1:
            raise ValueError("covariate_dependent censoring requires a binary event classifier with classes {0, 1}.")
        p_event = np.clip(event_proba[:, event_class_idx[0]], 0.0, 1.0)
        return np.random.binomial(1, p_event).astype('float32')
    if censoring_strategy == 'competing_times':
        raise RuntimeError(
            "'competing_times' is generated jointly by SurvivalPFN and should not be passed to _sample_synthetic_events()."
        )
    # 'random': draw at the real censoring rate, independent of covariates.
    return (np.random.rand(len(cov_gen_df)) >= censoring_ratio).astype('float32')


def _run_survival_function(df, columns, target_column, time_to_event_column, cat_indices,
                           n_generated_dataset, n_generated_sample, params, apply_rounding, feat_types_dict):
    """
    mode="survival_function": TabPFN generates covariates (plus the event column only when `censoring_strategy="joint"`); 
    the event indicator is set per `censoring_strategy`; and time is computed by a censoring-aware TTE model via `predict_any(X, event)`. 
    See the module docstring for the full rationale.
    """
    Xcov = df.drop(columns=[target_column, time_to_event_column])
    T = df[time_to_event_column]
    E = df[target_column]
    censoring_ratio = float((E == 0).mean())

    censoring_strategy = params.get('censoring_strategy', 'random')
    if censoring_strategy not in _CENSORING_STRATEGIES:
        raise ValueError(f"Unknown censoring_strategy {censoring_strategy!r}, expected one of {_CENSORING_STRATEGIES}")

    tte_method = params.get('tte_model', 'survival_function_regression')
    if censoring_strategy == 'competing_times' and tte_method != 'survivalpfn':
        raise ValueError(
            "censoring_strategy='competing_times' is only available with "
            "tte_model='survivalpfn'."
        )

    tte_model = _fit_tte_model(Xcov, T, E, method=tte_method, params=params,)

    censoring_clf = None
    if censoring_strategy == 'covariate_dependent':
        censoring_clf = TabPFNClassifier(device=params.get('device', 'auto'))
        censoring_clf.fit(Xcov, E)

    # Covariates-only joint fit (+ the event column for censoring_strategy="joint").
    cov_columns = list(Xcov.columns)
    fit_columns = cov_columns + [target_column] if censoring_strategy == 'joint' else cov_columns
    fit_cat_indices = [fit_columns.index(columns[i]) for i in cat_indices if columns[i] in fit_columns]
    model = _build_unsupervised_model(params)
    model.set_categorical_features(fit_cat_indices)
    model.fit(df[fit_columns])

    est_data_gen = []
    for _ in range(n_generated_dataset):
        gen_df = pd.DataFrame(_generate_once(model, n_generated_sample, params).numpy(), columns=fit_columns)
        cov_gen_df = gen_df[cov_columns].copy()

        if censoring_strategy == 'competing_times':
            # SurvivalPFN generates the survival pair coherently: draw latent
            # event and censoring times, then derive observed T and E from their
            # competition. No separate event sampler is used in this branch.
            time_syn, event_syn = tte_model.sample_observed(cov_gen_df)
        else:
            event_syn = _sample_synthetic_events(censoring_strategy, gen_df, cov_gen_df, target_column, censoring_clf, censoring_ratio,)
            event_syn_series = pd.Series(event_syn, index=cov_gen_df.index)
            time_syn = np.asarray(tte_model.predict_any(cov_gen_df, event_syn_series), dtype='float32')

        time_syn = np.clip(np.asarray(time_syn, dtype='float32'), a_min=_MIN_TIME, a_max=None)
        event_syn = np.asarray(event_syn, dtype='float32')

        out_df = cov_gen_df.copy()
        out_df[time_to_event_column] = time_syn
        out_df[target_column] = event_syn
        out_df = out_df[columns]  # restore the original column order
        out = torch.from_numpy(out_df.to_numpy(dtype='float32'))
        if apply_rounding:
            out = data_processing.round_data_gen(df.values, out, feat_types_dict)
        est_data_gen.append(out)
    return est_data_gen


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
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
        Shared generation knobs: `t` (temperature, default 1.0),
        `n_permutations` (default 3), `n_estimators` / `device` (TabPFN model
        knobs).

        mode="survival_function" only:
          - `tte_model` : time-to-event backend --
            "survival_function_regression" (default), "weibull_aft" or
            "survivalpfn".
          - `censoring_strategy` : "random" (default, matching the real
            censoring rate), "covariate_dependent" (Bernoulli-sampled from
            probabilities estimated by a fitted classifier), "joint" (event
            column generated jointly with covariates by TabPFN), or
            "competing_times" (SurvivalPFN only: sample latent event/censoring
            times and derive the observed `(T, E)` pair).
          - SurvivalPFN backend knobs (when `tte_model="survivalpfn"`):
            `survivalpfn_device`, `survivalpfn_model_path`,
            `survivalpfn_model_censoring` (bool), `survivalpfn_calibrate` (bool),
            `survivalpfn_sample_times` (bool, default True).
            `survivalpfn_point_estimate` ("median"/"mode"/"rmst") is used only
            when `survivalpfn_sample_times=False`.
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
    mode : {"naive", "survival_function"}, default="naive"
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
    if mode not in _MODES:
        raise ValueError(f"Unknown mode {mode!r}, expected one of {_MODES}")

    set_seed()
    params = params or {}

    df = pd.DataFrame(data.numpy(), columns=columns)  # Preprocessed dataset
    if n_generated_sample is None:
        n_generated_sample = df.shape[0]
    cat_indices = _categorical_indices(columns, target_column, feat_types_dict)

    runner = _run_naive if mode == 'naive' else _run_survival_function
    return runner(df, columns, target_column, time_to_event_column, cat_indices,
                  n_generated_dataset, n_generated_sample, params, apply_rounding, feat_types_dict)
