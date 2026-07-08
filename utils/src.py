#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Converted to PyTorch

Created on Mon Feb 17 16:17:14 2025

@author: Van Tuan NGUYEN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import random
import numpy as np

import likelihood, statistic, data_processing, theta_estimation

from diffuse_model import LatentDiffusion
import jax.numpy as jnp

def set_seed(seed=1):
    random.seed(seed)                            # Python built-in
    np.random.seed(seed)                         # NumPy
    torch.manual_seed(seed)                      # PyTorch (CPU)
    torch.cuda.manual_seed_all(seed)
    # Set deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

class _SurvPiecewiseDict(nn.Module):
    """Wraps piecewise-survival sub-layers so they are properly registered as
    submodules, while also carrying the (non-learnable) interval boundaries."""
    def __init__(self, theta_T, theta_C, intervals):
        super().__init__()
        self.theta_T = theta_T
        self.theta_C = theta_C
        self.intervals = intervals  # plain list, not a learnable param

    def __getitem__(self, key):
        return getattr(self, key)


class HIVAE(nn.Module):
    # Builds the encoder / decoder layers for a heterogeneous-incomplete VAE.
    # `self.theta_layer` is an `nn.ModuleDict` whose entries are themselves
    # `nn.ModuleDict` (or, for piecewise-survival features, the
    # `_SurvPiecewiseDict` helper) so that all sub-Linears are registered
    # as submodules and picked up by `vae_model.parameters()`.
    # `self.feat_normalization_globals` holds per-feature frozen statistics
    # (mean/var for real & pos, log mean/var for legacy surv, min/max for
    # the parametric survival families, None for cat/ordinal/count) so the
    # same scale is used by every forward pass during training, validation,
    # and generation. It is set by the train_HIVAE* functions on the full
    # training set before training, and remains None for inference paths
    # that don't need it (in which case data_processing.normalize_features
    # falls back to per-batch statistics).
    # Seeding is left to the caller (run / run_alt) — no global seed is set
    # in the constructor.
    def __init__(self, input_dim, z_dim, s_dim, y_dim, y_dim_partition=[], feat_types_dict=[], intervals_surv_piecewise=None, n_layers_surv_piecewise=2):

        super().__init__()
        self.feat_types_list = feat_types_dict

        # Determine Y dimensionality
        if y_dim_partition:
            self.y_dim_partition = y_dim_partition
        else:
            self.y_dim_partition = [y_dim] * len(self.feat_types_list)

        self.x_dim = input_dim
        self.z_dim = z_dim
        self.s_dim = s_dim
        self.y_dim = sum(self.y_dim_partition)
        self.feat_normalization_globals = None # populated later by train_HIVAE*

        # for encoder
        self.s_layer = nn.Linear(input_dim, s_dim)  # q(s|x^o)
        self.z_layer = nn.Linear(input_dim + s_dim, z_dim * 2)  # q(z|s,x^o)

        # for decoder
        self.z_distribution_layer = nn.Linear(s_dim, z_dim)  # p(z|s)
        self.y_layer = nn.Linear(z_dim, self.y_dim)  # y deterministic layer
        
        self.theta_layer = nn.ModuleDict()
        for i, feat in enumerate(self.feat_types_list):

            feat_y_dim = self.y_dim_partition[i]

            if feat['type'] in ['real', 'pos']:
                self.theta_layer["feat_" + str(i)] = nn.ModuleDict({
                    'mean' : nn.Linear(feat_y_dim + s_dim, 1, bias=False),
                    'sigma' : nn.Linear(s_dim, 1, bias=False)
                })
            elif feat['type'] in ['surv']:
                self.theta_layer["feat_" + str(i)] = nn.ModuleDict({
                    'mean_T' : nn.Linear(feat_y_dim + s_dim, 1, bias=False),
                    'sigma_T' : nn.Linear(s_dim, 1, bias=False),
                    'mean_C' : nn.Linear(feat_y_dim + s_dim, 1, bias=False),
                    'sigma_C' : nn.Linear(s_dim, 1, bias=False)
                })
            elif feat['type'] in ['surv_weibull','surv_loglog']:
                self.theta_layer["feat_" + str(i)] = nn.ModuleDict({
                    'theta' : nn.Linear(feat_y_dim + s_dim, 4, bias=False)
                })
            elif feat['type'] in ['surv_piecewise']:
                n_intervals = len(intervals_surv_piecewise)
                if n_layers_surv_piecewise == 2:
                    self.theta_layer["feat_" + str(i)] = _SurvPiecewiseDict(
                        theta_T=nn.Sequential(
                            nn.Linear(feat_y_dim + s_dim, out_features=20, bias=False),
                            nn.ReLU(),
                            nn.Linear(in_features=20, out_features=n_intervals, bias=False)
                        ),
                        theta_C=nn.Sequential(
                            nn.Linear(feat_y_dim + s_dim, out_features=20, bias=False),
                            nn.ReLU(),
                            nn.Linear(in_features=20, out_features=n_intervals, bias=False)
                        ),
                        intervals=intervals_surv_piecewise
                    )
                else:
                    self.theta_layer["feat_" + str(i)] = _SurvPiecewiseDict(
                        theta_T=nn.Linear(feat_y_dim + s_dim, n_intervals, bias=False),
                        theta_C=nn.Linear(feat_y_dim + s_dim, n_intervals, bias=False),
                        intervals=intervals_surv_piecewise
                    )
            elif feat['type'] in ['count']:
                self.theta_layer["feat_" + str(i)] = nn.Linear(feat_y_dim + s_dim, 1, bias=False)
            elif feat['type'] in ['cat']:
                n_class = int(feat['nclass'])
                self.theta_layer["feat_" + str(i)] = nn.Linear(feat_y_dim + s_dim, n_class - 1, bias=False)
            else: # ordinal
                n_class = int(feat['nclass'])
                self.theta_layer["feat_" + str(i)] = nn.ModuleDict({
                    'theta' : nn.Linear(s_dim, n_class - 1, bias=False),
                    'mean' : nn.Linear(feat_y_dim + s_dim, 1, bias=False)
                })


    def forward(self, batch_data_oberved, batch_data, batch_miss, tau=1.0, n_generated_dataset=1):
        """
        Forward pass through the encoder and decoder.

        Input features are normalized using the per-feature frozen statistics
        stored in `self.feat_normalization_globals` when it has been set
        (typically by the train_HIVAE* functions, which freeze it on the full
        training set before training starts). When it is None,
        normalize_features falls back to per-batch statistics.
        """

        # Feature normalization (frozen training-set stats when available)
        X_list, normalization_params = data_processing.normalize_features(
            batch_data_oberved, self.feat_types_list, batch_miss,
            feat_normalization_globals=self.feat_normalization_globals,
        )
        
        # Encode
        X = torch.cat(X_list, dim=1) 
        q_params, samples = self.encode(X, tau)
        
        # Decode
        p_params, log_p_x, log_p_x_missing, samples = self.decode(samples, batch_data, batch_miss, normalization_params, n_generated_dataset)

        # Compute loss
        ELBO, loss_reconstruction, KL_z, KL_s = self.loss_function(log_p_x, p_params, q_params)

        return {
            "samples": samples,
            "log_p_x": log_p_x,
            "log_p_x_missing": log_p_x_missing,
            "loss_re": loss_reconstruction,
            "neg_ELBO_loss": -ELBO,
            "KL_s": KL_s,
            "KL_z": KL_z,
            "p_params": p_params,
            "q_params": q_params
        }

    def diffusion_forward(
        self,
        batch_data_observed,
        batch_data,
        batch_miss,
        tau: float = 1.0,
        n_generated_dataset: int = 1,
        diffusion_hidden_dim: int = 32,
        diffusion_n_steps: int = 100,
        diffusion_n_epochs: int = 200,
        diffusion_batch_size: int = 2000,
        diffusion_lr: float = 1e-3,
    ):
        """
        Encode → fit latent diffusion model → sample → decode.

        The score network is trained from scratch on the encoded latents of
        the current batch, then used to draw new latent samples that are
        decoded exactly like the standard forward pass.
        """
        # ── 1. Normalize & encode ──────────────────────────────────────
        X_list, normalization_params = data_processing.normalize_features(
            batch_data_observed, self.feat_types_list, batch_miss,
            feat_normalization_globals=self.feat_normalization_globals,
        )
        X = torch.cat(X_list, dim=1)
        q_params, samples = self.encode(X, tau)

        # ── 2. Fit diffusion on encoded latents ───────────────────────
        latent_dim = self.z_dim
        latents_np = jnp.array(samples["z"].detach().numpy())
        n_generated_sample = samples["z"].shape[0]

        diffusion = LatentDiffusion(
            latent_dim=latent_dim,
            hidden_dim=diffusion_hidden_dim,
            n_steps=diffusion_n_steps,
            n_epochs=diffusion_n_epochs,
            batch_size=diffusion_batch_size,
            lr=diffusion_lr,
        )
        diffusion.fit(latents_np)
        # diffusion.plot_convergence()

        # ── 3. Sample new latents & convert to torch ──────────────────
        diffusion_samples = diffusion.sample(n_generated_sample)
        samples["z"] = torch.tensor(np.array(diffusion_samples), dtype=torch.float32)

        # ── 4. Decode & compute loss ──────────────────────────────────
        p_params, log_p_x, log_p_x_missing, samples = self.decode(
            samples, batch_data, batch_miss, normalization_params, n_generated_dataset
        )
        ELBO, loss_reconstruction, KL_z, KL_s = self.loss_function(log_p_x, p_params, q_params)

        return {
            "samples": samples,
            "log_p_x": log_p_x,
            "log_p_x_missing": log_p_x_missing,
            "loss_re": loss_reconstruction,
            "neg_ELBO_loss": -ELBO,
            "KL_s": KL_s,
            "KL_z": KL_z,
            "p_params": p_params,
            "q_params": q_params,
        }
    

    def diffusion_forward_y(
        self,
        batch_data_observed,
        batch_data,
        batch_miss,
        tau: float = 1.0,
        n_generated_dataset: int = 1,
        diffusion_hidden_dim: int = 32,
        diffusion_n_steps: int = 100,
        diffusion_n_epochs: int = 200,
        diffusion_batch_size: int = 2000,
        diffusion_lr: float = 1e-3,
    ):
        """
        Encode → fit latent diffusion model → sample → decode.

        The score network is trained from scratch on the encoded latents of
        the current batch, then used to draw new latent samples that are
        decoded exactly like the standard forward pass.
        """
        # ── 1. Normalize & encode ──────────────────────────────────────
        X_list, normalization_params = data_processing.normalize_features(
            batch_data_observed, self.feat_types_list, batch_miss,
            feat_normalization_globals=self.feat_normalization_globals,
        )
        X = torch.cat(X_list, dim=1)
        q_params, samples = self.encode(X, tau)

        # ── 2. Fit diffusion on encoded latents ───────────────────────
        latent_dim = self.y_dim
        latents_np = jnp.array(samples["y"].detach().numpy())
        n_generated_sample = samples["y"].shape[0]

        diffusion = LatentDiffusion(
            latent_dim=latent_dim,
            hidden_dim=diffusion_hidden_dim,
            n_steps=diffusion_n_steps,
            n_epochs=diffusion_n_epochs,
            batch_size=diffusion_batch_size,
            lr=diffusion_lr,
        )
        diffusion.fit(latents_np)
        # diffusion.plot_convergence()

        # ── 3. Sample new latents & convert to torch ──────────────────
        diffusion_samples = diffusion.sample(n_generated_sample)
        samples["y"] = torch.tensor(np.array(diffusion_samples), dtype=torch.float32)

        # ── 4. Decode & compute loss ──────────────────────────────────
        # p_params, log_p_x, log_p_x_missing, samples = self.decode(
        #     samples, batch_data, batch_miss, normalization_params, n_generated_dataset
        # )
        grouped_samples_y = data_processing.y_partition(samples["y"], self.feat_types_list, self.y_dim_partition)
        # Compute θ parameters    
        theta = theta_estimation.theta_estimation_from_ys(grouped_samples_y, samples["s"], self.feat_types_list, batch_miss, self.theta_layer)
        # Compute log-likelihood and reconstructed data
        _, _, _, samples["x"] = likelihood.loglik_evaluation(
            batch_data, self.feat_types_list, batch_miss, theta, normalization_params, n_generated_dataset
        )
        # ELBO, loss_reconstruction, KL_z, KL_s = self.loss_function(log_p_x, p_params, q_params)

        return {
            "samples": samples
        }

    def decode(self, samples, batch_data_list, miss_list, normalization_params, n_generated_dataset=1):
        """
        Decodes latent variables into output reconstructions.

        Parameters:
        -----------
        samples : dict
            Sampled latent variables {s, z}.
        
        batch_data_list : list of torch.Tensor
            Original batch data.
        
        miss_list : torch.Tensor
            Mask indicating missing data.
        
        normalization_params : dict
            Normalization parameters for data.

        Returns:
        --------
        
        p_params : dict
            Parameters of the prior distributions.
        
        log_p_x : torch.Tensor
            Log-likelihood of observed data.
        
        log_p_x_missing : torch.Tensor
            Log-likelihood of missing data.

        samples : dict
            Updated dictionary containing generated samples.
        """
        p_params = {}

        # Compute p(z|s)
        mean_pz, log_var_pz = statistic.z_prior_GMM(samples["s"], self.z_distribution_layer)
        p_params["z"] = (mean_pz, log_var_pz)

        # Compute deterministic y layer
        samples["y"] = self.y_layer(samples["z"])

        # Partition y
        grouped_samples_y = data_processing.y_partition(samples["y"], self.feat_types_list, self.y_dim_partition)

        # Compute θ parameters    
        theta = theta_estimation.theta_estimation_from_ys(grouped_samples_y, samples["s"], self.feat_types_list, miss_list, self.theta_layer)

        # Compute log-likelihood and reconstructed data
        p_params["x"], log_p_x, log_p_x_missing, samples["x"] = likelihood.loglik_evaluation(
            batch_data_list, self.feat_types_list, miss_list, theta, normalization_params, n_generated_dataset
        )
        return p_params, log_p_x, log_p_x_missing, samples


    def loss_function(self, log_p_x, p_params, q_params):
        """
        Computes the Evidence Lower Bound (ELBO) for the Variational Autoencoder.

        Parameters:
        -----------
        log_p_x : torch.Tensor
            Log-likelihood of reconstructed samples.
        
        p_params : dict
            Parameters of prior distributions.
        
        q_params : dict
            Parameters of variational distributions.

        Returns:
        --------
        ELBO : torch.Tensor
            Evidence Lower Bound loss.
        
        loss_reconstruction : torch.Tensor
            Reconstruction loss term.
        
        KL_z : torch.Tensor
            KL divergence for z.
        
        KL_s : torch.Tensor
            KL divergence for s. Per-sample (shape [batch]); the batch mean
            is taken when assembling the ELBO, so cross_entropy is called
            with reduction='none' here.
        """

        # KL(q(s|x) || p(s))
        log_pi = q_params['s']
        pi_param = F.softmax(log_pi, dim=-1)
        KL_s = -F.cross_entropy(log_pi, pi_param, reduction='none') + torch.log(torch.tensor(float(self.s_dim)))

        # KL(q(z|s,x) || p(z|s))
        mean_pz, log_var_pz = p_params['z']
        mean_qz, log_var_qz = q_params['z']
        
        KL_z = -0.5 * self.z_dim + 0.5 * torch.sum(
            torch.exp(log_var_qz - log_var_pz) + (mean_pz - mean_qz).pow(2) / torch.exp(log_var_pz) - log_var_qz + log_var_pz, dim=1
        )
        # Expectation of log p(x|y)
        loss_reconstruction = torch.sum(log_p_x, dim=0)

        # Complete ELBO
        ELBO = torch.mean(loss_reconstruction - KL_z - KL_s, dim=0)

        return ELBO, loss_reconstruction, KL_z, KL_s


class HIVAE_factorized(HIVAE):

    """
    The HI_VAE Model with factorized encoder.

    This model encodes input data into latent variables (s, z) using variational inference,
    and decodes these representations to reconstruct the original input.

    Parameters:
    -----------
    input_dim : int
        Dimensionality of input data.
    
    z_dim : int
        Dimensionality of the latent variable z.
    
    s_dim : int
        Number of categorical latent states (s).

    y_dim : int
        Dimensionality of the deterministic layer y.

    y_dim_partition : list
        Partitioning dimensions for input variables.

    feat_types_file : str
        
    """

    def __init__(self, input_dim, z_dim, s_dim, y_dim, y_dim_partition, feat_types_dict, intervals_surv_piecewise, n_layers_surv_piecewise=2):

        # print(f'[*] Importing model: {model_name}')
        super().__init__(input_dim, z_dim, s_dim, y_dim, y_dim_partition, feat_types_dict, intervals_surv_piecewise, n_layers_surv_piecewise)
    
    def encode(self, X, tau):
        """
        Encodes input data X into latent variables s and z using variational inference.

        Parameters:
        -----------
        X : torch.Tensor
            Input data batch.
        tau : float
            Temperature parameter for Gumbel-softmax.

        Returns:
        --------        
        q_params : dict
            Parameters of the variational distributions {s_logits, (mean_qz, log_var_qz)}.

        samples : dict
            Sampled latent variables {s, z}.
        """

        # Softmax over s (categorical distribution)
        logits_s = self.s_layer(X)
        p_s = F.softmax(logits_s, dim=-1)

        # Gumbel-softmax trick
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(p_s)))
        samples_s = F.softmax(torch.log(torch.clamp(p_s, 1e-6, 1)) + gumbel_noise / tau, dim=-1)

        # Compute q(z|s,x^o)
        z_params = self.z_layer(torch.cat([X, samples_s], dim=1))
        mean_qz, log_var_qz = torch.chunk(z_params, 2, dim=1)
        log_var_qz = torch.clamp(log_var_qz, -15.0, 15.0)

        # Reparametrization trick
        eps = torch.randn_like(mean_qz)
        samples_z = mean_qz + torch.exp(log_var_qz / 2) * eps

        q_params = {"s": logits_s, "z": (mean_qz, log_var_qz)} 
        samples = {"s": samples_s, "z": samples_z}

        return q_params, samples



class HIVAE_inputDropout(HIVAE):

    """
    The HI_VAE model with input dropout encoder.

    This model encodes input data into latent variables (s, z) using variational inference,
    and decodes these representations to reconstruct the original input.

    Parameters:
    -----------
    input_dim : int
        Dimensionality of input data.
    
    z_dim : int
        Dimensionality of the latent variable z.
    
    s_dim : int
        Number of categorical latent states (s).
    
    y_dim : int
        Dimensionality of the deterministic layer y.

    y_dim_partition : list
        Partitioning dimensions for input variables.

    feat_types_file : str
        
    """

    def __init__(self, input_dim, z_dim, s_dim, y_dim, y_dim_partition, feat_types_dict, intervals_surv_piecewise, n_layers_surv_piecewise=2):

        # print(f'[*] Importing model: {model_name}')
        super().__init__(input_dim, z_dim, s_dim, y_dim, y_dim_partition, feat_types_dict, intervals_surv_piecewise, n_layers_surv_piecewise)
    
    def encode(self, X, tau):
        """
        Encodes input data X into latent variables s and z using variational inference.

        Parameters:
        -----------
        X : torch.Tensor
            Input data batch.
        tau : float
            Temperature parameter for Gumbel-softmax.

        Returns:
        --------        
        q_params : dict
            Parameters of the variational distributions {s_logits, (mean_qz, log_var_qz)}.

        samples : dict
            Sampled latent variables {s, z}.
        """

        #Create the proposal of q(s|x^o)
        samples_s, s_params = statistic.s_proposal_multinomial(X, self.s_layer, tau)

        # Compute q(z|s,x^o)
        batch_size = X.shape[0]
        samples_z, z_params = statistic.z_proposal_GMM(X, samples_s, batch_size, self.z_dim, self.z_layer)

        q_params = {"s": s_params, "z": z_params} 
        samples = {"s": samples_s, "z": samples_z}

        return q_params, samples