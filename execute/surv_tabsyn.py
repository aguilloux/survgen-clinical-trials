"""Naive joint TabSyn adapter using the official amazon-science/tabsyn models.

Set params['tabsyn_root'] to an upstream checkout (see script/simulation_TabSyn.md).
Time is numerical; the event indicator is categorical. No survival likelihood.
"""
from copy import deepcopy
import hashlib
import importlib
from numbers import Integral
from pathlib import Path
import random
import sys
from time import perf_counter

import numpy as np
import torch
from sklearn.preprocessing import QuantileTransformer
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

UPSTREAM_FILES = ('tabsyn/vae/model.py', 'tabsyn/model.py', 'tabsyn/diffusion_utils.py')


def source_fingerprint(root):
    root = Path(root).expanduser().resolve()
    for name in UPSTREAM_FILES:
        if not (root / name).is_file():
            raise FileNotFoundError(f'Missing {root / name}; provide an official TabSyn checkout.')
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in UPSTREAM_FILES}


def _backend(root):
    root = Path(root).expanduser().resolve()
    source_fingerprint(root)
    # Upstream model modules only depend on torch/numpy/scipy, not its CLI utilities.
    sys.path.insert(0, str(root))
    try:
        modules = [importlib.import_module(name) for name in
                   ('tabsyn.vae.model', 'tabsyn.model', 'tabsyn.diffusion_utils')]
        if any(root not in Path(module.__file__).resolve().parents for module in modules):
            raise RuntimeError('A different TabSyn checkout is already imported; use a fresh process.')
        return modules
    finally:
        sys.path.remove(str(root))


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _positive(name, value, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}.')
    return int(value)


def run(data, columns, target_column, time_to_event_column, n_generated_dataset,
        n_generated_sample=None, params=None, cond_gen=None, apply_rounding=False,
        feat_types_dict=None, mode='naive', profile=None):
    """Fit a transformer VAE, fit latent EDM, then decode independent seeded draws.

    Uses upstream architectures, loss and sampler; training orchestration is local.
    Model selection and beta annealing use training reconstruction loss (MSE + CE),
    never the trial's treated arm. See the accompanying notes for differences from
    the upstream benchmark runner. Conditional generation is not implemented.
    """
    if mode != 'naive' or cond_gen is not None:
        raise ValueError('TabSyn supports unconditional naive generation only.')
    columns = list(columns)
    if len(set(columns)) != len(columns):
        raise ValueError('Column names must be unique.')
    if target_column == time_to_event_column or not {target_column, time_to_event_column} <= set(columns):
        raise ValueError('Specify distinct event/time columns present in columns.')
    values = data.detach().cpu().numpy() if torch.is_tensor(data) else np.asarray(data)
    if values.ndim != 2 or values.shape[1] != len(columns) or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('Training data must be finite, match columns, and contain at least two rows.')
    event_idx, time_idx = columns.index(target_column), columns.index(time_to_event_column)
    if not np.isin(values[:, event_idx], [0, 1]).all() or not (values[:, time_idx] > 0).all():
        raise ValueError('Event must be binary and observed time strictly positive.')
    if feat_types_dict is None:
        raise ValueError('feat_types_dict is required to identify categorical covariates.')
    categorical = {event_idx}
    offset = 0
    for feature in feat_types_dict:
        if feature['type'] in ('cat', 'ordinal'):
            categorical.add(offset)
        offset += 2 if feature['type'] in ('surv', 'surv_weibull', 'surv_piecewise') else 1
    if offset != len(columns) or time_idx in categorical:
        raise ValueError('Feature descriptors must match columns; time must be numerical.')
    cat_idx = sorted(categorical)
    num_idx = [i for i in range(len(columns)) if i not in categorical]
    count = _positive('n_generated_dataset', n_generated_dataset)
    size = _positive('n_generated_sample', len(values) if n_generated_sample is None else n_generated_sample)
    options = dict(params or {})
    root = options.pop('tabsyn_root', None)
    if root is None:
        raise ValueError('tabsyn_root must point to an official TabSyn checkout.')
    expected_source = options.pop('source_sha256', None)
    fingerprint = source_fingerprint(root)
    if expected_source is not None and fingerprint != expected_source:
        raise ValueError('TabSyn source changed since experiment configuration was created.')
    seed = _positive('random_state', options.pop('random_state', 1), minimum=0)
    vae_epochs = _positive('vae_epochs', options.pop('vae_epochs', 4000))
    diffusion_epochs = _positive('diffusion_epochs', options.pop('diffusion_epochs', 10001))
    batch_size = _positive('batch_size', options.pop('batch_size', 4096))
    steps = _positive('sampling_steps', options.pop('sampling_steps', 50), minimum=2)
    width = _positive('denoiser_width', options.pop('denoiser_width', 1024), minimum=4)
    if width % 2:
        raise ValueError('denoiser_width must be even.')
    device = options.pop('device', 'auto')
    if options:
        raise ValueError(f'Unknown TabSyn parameters: {sorted(options)}')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable.')
    vae_module, diffusion_module, sampler = _backend(root)
    _seed(seed)
    profile = {} if profile is None else profile
    profile['source_sha256'] = fingerprint
    profile['device'] = str(device)
    started = perf_counter()
    transform = QuantileTransformer(n_quantiles=min(len(values), 1000),
                                    output_distribution='normal', random_state=seed)
    numerical = torch.tensor(transform.fit_transform(values[:, num_idx]), dtype=torch.float32)
    levels = [np.unique(values[:, i]) for i in cat_idx]
    categories = torch.tensor(np.column_stack([
        np.searchsorted(level, values[:, i]) for level, i in zip(levels, cat_idx)]), dtype=torch.long)
    architecture = (2, len(num_idx), [len(level) for level in levels], 4)
    vae = vae_module.Model_VAE(*architecture, n_head=1, factor=32, bias=True).to(device)
    loader = DataLoader(TensorDataset(numerical, categories), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.95, patience=10)
    beta, best_loss, patience = 1e-2, float('inf'), 0
    best_state = None
    for epoch in range(vae_epochs):
        vae.train()
        total = 0.
        for x_num, x_cat in loader:
            x_num, x_cat = x_num.to(device), x_cat.to(device)
            recon_num, recon_cat, mu, logvar = vae(x_num, x_cat)
            reconstruction = F.mse_loss(recon_num, x_num) + torch.stack([
                F.cross_entropy(logits, x_cat[:, i]) for i, logits in enumerate(recon_cat)]).mean()
            kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).mean()
            loss = reconstruction + beta * kl
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite TabSyn VAE loss.')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += reconstruction.item() * len(x_num)
        metric = total / len(values)
        scheduler.step(metric)
        if metric < best_loss:
            best_loss, best_state, patience = metric, deepcopy(vae.state_dict()), 0
        else:
            patience += 1
            if patience % 10 == 0:
                beta = max(1e-5, beta * 0.7)
        if epoch == 0 or (epoch + 1) % 100 == 0:
            print(f'TabSyn VAE epoch {epoch + 1}/{vae_epochs}: reconstruction={metric:.6g}, beta={beta:.3g}', flush=True)
    vae.load_state_dict(best_state)
    encoder = vae_module.Encoder_model(*architecture, n_head=1, factor=32).to(device)
    decoder = vae_module.Decoder_model(*architecture, n_head=1, factor=32).to(device)
    encoder.load_weights(vae)
    decoder.load_weights(vae)
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        latent = torch.cat([encoder(n.to(device), c.to(device))[:, 1:].flatten(1).cpu()
                            for n, c in DataLoader(TensorDataset(numerical, categories), batch_size=batch_size)])
    mean = latent.mean(0)
    latent = (latent - mean) / 2  # Upstream TabSyn normalization, not per-column standardization.
    profile.update(vae_seconds=perf_counter() - started, vae_best_reconstruction=best_loss,
                   vae_epochs=vae_epochs, final_beta=beta)
    del vae, encoder, best_state
    started = perf_counter()
    diffusion = diffusion_module.Model(
        denoise_fn=diffusion_module.MLPDiffusion(latent.shape[1], width),
        hid_dim=latent.shape[1]).to(device)
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=20)
    best_loss, patience, best_state = float('inf'), 0, None
    for epoch in range(diffusion_epochs):
        diffusion.train()
        total = 0.
        for batch in DataLoader(latent, batch_size=batch_size, shuffle=True):
            loss = diffusion(batch.to(device)).mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite TabSyn diffusion loss.')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        metric = total / len(latent)
        scheduler.step(metric)
        if metric < best_loss:
            best_loss, best_state, patience = metric, deepcopy(diffusion.state_dict()), 0
        else:
            patience += 1
        if epoch == 0 or (epoch + 1) % 100 == 0:
            print(f'TabSyn diffusion epoch {epoch + 1}/{diffusion_epochs}: loss={metric:.6g}', flush=True)
        if patience >= 500:
            break
    diffusion.load_state_dict(best_state)
    diffusion.eval()
    profile.update(diffusion_seconds=perf_counter() - started, diffusion_best_loss=best_loss,
                   diffusion_epochs=epoch + 1)
    if apply_rounding:
        try:
            from utils import data_processing
        except ModuleNotFoundError:
            import data_processing
        rounding_plan = data_processing.prepare_rounding_plan(values, feat_types_dict)
    started = perf_counter()
    results = []
    for draw in range(count):
        _seed(seed + draw + 1)
        chunks = []
        with torch.no_grad():
            for start in range(0, size, batch_size):
                n = min(batch_size, size - start)
                z = sampler.sample(diffusion.denoise_fn_D, n, latent.shape[1], num_steps=steps, device=device)
                z = (z * 2 + mean.to(device)).reshape(n, len(columns), 4)
                num_out, cat_out = decoder(z)
                out = np.empty((n, len(columns)), dtype=np.float32)
                out[:, num_idx] = transform.inverse_transform(num_out.cpu().numpy())
                for i, logits, level in zip(cat_idx, cat_out, levels):
                    out[:, i] = level[logits.argmax(-1).cpu().numpy()]
                chunks.append(torch.from_numpy(out))
        out = torch.cat(chunks)
        if not torch.isfinite(out).all():
            raise RuntimeError('TabSyn returned non-finite samples.')
        if apply_rounding:
            out = data_processing.round_data_gen(values, out, feat_types_dict, rounding_plan=rounding_plan)
        out[:, time_idx].clamp_(min=1e-3)
        results.append(out)
    profile['sampling_seconds'] = perf_counter() - started
    return results
